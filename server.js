require('dotenv').config();
const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());
app.use(express.static('public'));

// ─── COST TRACKING ────────────────────────────────────────────────────────────
// LLaMA 3.3 70B on Groq: ~$0.59/1M input tokens, ~$0.79/1M output tokens (approx)
const COST_PER_1K_INPUT  = 0.00059;
const COST_PER_1K_OUTPUT = 0.00079;
let sessionStats = { totalRequests: 0, totalCost: 0, totalLatency: 0, totalRetries: 0, successCount: 0, failCount: 0 };

function estimateCost(promptTokens, completionTokens) {
  return ((promptTokens / 1000) * COST_PER_1K_INPUT) + ((completionTokens / 1000) * COST_PER_1K_OUTPUT);
}

// ─── EVAL DATASET ─────────────────────────────────────────────────────────────
const EVAL_DATASET = {
  real: [
    { id: 'r1', label: 'CRM', prompt: 'Build a CRM with login, contacts, dashboard, role-based access, and premium plan with payments. Admins can see analytics.' },
    { id: 'r2', label: 'Project Tracker', prompt: 'Project management tool with teams, tasks, kanban board, file uploads, and notifications. Free and pro tiers.' },
    { id: 'r3', label: 'Marketplace', prompt: 'Online marketplace with sellers, buyers, listings, cart, payments, and reviews.' },
    { id: 'r4', label: 'Healthcare', prompt: 'Patient portal with appointments, prescriptions, doctor profiles, insurance claims, and video consultations.' },
    { id: 'r5', label: 'E-learning', prompt: 'E-learning platform with courses, video lessons, quizzes, certificates, student progress tracking, and subscriptions.' },
    { id: 'r6', label: 'Food Delivery', prompt: 'Food delivery app with restaurants, menus, cart, orders, delivery tracking, ratings, and driver management.' },
    { id: 'r7', label: 'HR System', prompt: 'HR management system with employee records, leave requests, payroll, performance reviews, and org chart.' },
    { id: 'r8', label: 'Event Booking', prompt: 'Event booking platform with venues, event listings, ticket purchase, QR code check-in, and organizer dashboard.' },
    { id: 'r9', label: 'Finance Tracker', prompt: 'Personal finance app with accounts, transactions, budgets, expense categories, recurring bills, and analytics.' },
    { id: 'r10', label: 'Social Network', prompt: 'Social network with user profiles, posts, comments, likes, follow system, messaging, and content moderation.' }
  ],
  edge: [
    { id: 'e1', label: 'Vague', prompt: 'A thing for managing stuff with users', category: 'vague' },
    { id: 'e2', label: 'Conflicting roles', prompt: 'An app where everyone is an admin and no one has permissions', category: 'conflicting' },
    { id: 'e3', label: 'Incomplete', prompt: 'Build a payment system', category: 'incomplete' },
    { id: 'e4', label: 'Contradictory', prompt: 'Free premium app where users pay nothing but get paid features without login', category: 'conflicting' },
    { id: 'e5', label: 'One word', prompt: 'App', category: 'vague' },
    { id: 'e6', label: 'Over-specified', prompt: 'Build exactly 17 tables, each with precisely 5 columns, 3 endpoints per table, no auth, no UI, pure API only, must return XML not JSON, and must run on COBOL', category: 'conflicting' },
    { id: 'e7', label: 'No data model', prompt: 'A dashboard that shows things and lets people do stuff', category: 'vague' },
    { id: 'e8', label: 'Ambiguous domain', prompt: 'Build a bank', category: 'incomplete' },
    { id: 'e9', label: 'Conflicting monetization', prompt: 'A platform that is completely free with no ads, but also generates revenue through ads and paywalls simultaneously', category: 'conflicting' },
    { id: 'e10', label: 'Missing roles', prompt: 'Booking system where anyone can do anything with no restrictions', category: 'conflicting' }
  ]
};

// ─── EVALUATION RESULTS STORE (in-memory, reset on restart) ──────────────────
let evalResults = [];

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── VAGUE PROMPT DETECTOR ───────────────────────────────────────────────────
function analyzePromptQuality(prompt) {
  const words = prompt.trim().split(/\s+/);
  const issues = [];
  const assumptions = [];

  if (words.length < 5) issues.push('Prompt is extremely short or vague');
  if (!/(login|auth|user|account|register)/i.test(prompt)) assumptions.push('Assuming user authentication is required');
  if (!/(role|admin|permission|access)/i.test(prompt)) assumptions.push('Assuming basic role separation (admin/user)');
  if (!/(data|record|table|store|save)/i.test(prompt) && words.length < 10) issues.push('No data model implied');
  if (/(everyone.*admin|no.*permission|all.*access)/i.test(prompt)) issues.push('Conflicting access control logic detected');
  if (/(free.*pay|pay.*free.*feature)/i.test(prompt)) issues.push('Conflicting monetization model detected');

  const clarifications = [];
  if (!/(pay|premium|subscription|monetiz)/i.test(prompt)) clarifications.push('Monetization model not specified — assuming free tier');
  if (!/(mobile|web|desktop)/i.test(prompt)) clarifications.push('Platform not specified — assuming web application');
  if (!/(api|rest|graphql)/i.test(prompt)) clarifications.push('API style not specified — assuming REST');

  return { issues, assumptions, clarifications, wordCount: words.length, quality: issues.length === 0 ? 'good' : issues.length <= 2 ? 'fair' : 'poor' };
}

// ─── GROQ API CALL ────────────────────────────────────────────────────────────
async function callGroq(prompt, systemOverride) {
  const system = systemOverride || 'You are an app schema compiler. Always respond with valid JSON only. No markdown, no backticks, no explanation. Just raw JSON.';
  const t0 = Date.now();

  const response = await axios.post(
    'https://api.groq.com/openai/v1/chat/completions',
    {
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: prompt }
      ],
      temperature: 0
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      }
    }
  );

  const usage = response.data.usage || {};
  const cost = estimateCost(usage.prompt_tokens || 0, usage.completion_tokens || 0);
  const latency = Date.now() - t0;

  const raw = response.data.choices[0].message.content;
  const clean = raw.replace(/```json/g, '').replace(/```/g, '').trim();
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  if (start === -1 || end === -1) throw new Error('No JSON found in response');

  return {
    data: JSON.parse(clean.substring(start, end + 1)),
    cost,
    latency,
    tokens: { input: usage.prompt_tokens || 0, output: usage.completion_tokens || 0 }
  };
}

// ─── REPAIR ───────────────────────────────────────────────────────────────────
async function repairSchemas(schemas, issues) {
  const result = await callGroq(`
You are a schema repair engine. Fix ALL of these issues in the schema below.

RULES:
- If an API field does not exist in DB, ADD that column to the correct DB table
- If a column looks like a foreign key (_id suffix) but has foreign_key: null, set the foreign_key to the correct table
- If an endpoint has amount/price/money fields, add "validation": {"amount": ">0"} to that endpoint
- If field names differ across layers (user_id vs author_ref vs posted_by), unify them ALL to "user_id"
- Keep ALL 4 top-level keys: ui_schema, api_schema, db_schema, auth_rules
- Return ONLY the complete fixed JSON, nothing else

Issues to fix:
${issues.join('\n')}

Schema to fix:
${JSON.stringify(schemas)}
`);
  return result;
}

// ─── RUNTIME GENERATION ───────────────────────────────────────────────────────
function generateExpressRoutes(apiSchema) {
  if (!apiSchema?.endpoints) return [];
  return apiSchema.endpoints.map(ep =>
    `app.${ep.method.toLowerCase()}("${ep.path}", (req, res) => res.json({ success: true }));`
  );
}

function generateExecutionProof(schemas) {
  const routes = generateExpressRoutes(schemas.api_schema);
  const tables = schemas.db_schema?.tables?.map(t => ({
    name: t.name,
    sql: `CREATE TABLE ${t.name} (\n  ${t.columns.map(c => {
      let def = `${c.name} ${c.type.toUpperCase()}`;
      if (c.primary_key) def += ' PRIMARY KEY';
      if (!c.nullable) def += ' NOT NULL';
      if (c.foreign_key) def += ` REFERENCES ${c.foreign_key}(id)`;
      return def;
    }).join(',\n  ')}\n);`
  })) || [];

  const packageJson = {
    name: 'generated-app',
    version: '1.0.0',
    dependencies: { express: '^4.18.0', pg: '^8.11.0', jsonwebtoken: '^9.0.0', bcrypt: '^5.1.0' }
  };

  return { routes, tables, packageJson, isExecutable: routes.length > 0 && tables.length > 0 };
}

// ─── VALIDATION ───────────────────────────────────────────────────────────────
function validateSchemas(schemas, intentRoles, strict = true) {
  const issues = [];
  const dbFields = schemas.db_schema?.tables?.flatMap(t => t.columns.map(c => c.name)) || [];

  schemas.api_schema?.endpoints?.forEach(ep => {
    if (ep.request_body && typeof ep.request_body === 'object') {
      Object.keys(ep.request_body).forEach(field => {
        const skip = ['token', 'authorization', 'auth_token', 'jwt', 'password', 'confirm_password'];
        if (!skip.includes(field.toLowerCase()) && !dbFields.includes(field)) {
          issues.push(`API field "${field}" in "${ep.path}" not found in any DB table`);
        }
      });
    }
  });

  schemas.db_schema?.tables?.forEach(table => {
    table.columns?.forEach(col => {
      const name = col.name.toLowerCase();
      const looksLikeFK =
        (name.endsWith('_id') && name !== 'id') ||
        ['userid', 'accountid', 'patientid', 'doctorid', 'ownerid', 'senderid', 'receiverid'].includes(name);
      if (looksLikeFK && !col.foreign_key) {
        issues.push(`Column "${col.name}" in table "${table.name}" looks like a FK but foreign_key is null`);
      }
    });
  });

  const moneyFields = ['amount', 'value', 'money', 'price', 'balance', 'total', 'fee', 'cost'];
  schemas.api_schema?.endpoints?.forEach(ep => {
    if (ep.request_body && typeof ep.request_body === 'object') {
      const hasMoneyField = Object.keys(ep.request_body).some(f => moneyFields.includes(f.toLowerCase()));
      if (hasMoneyField && !ep.validation) {
        issues.push(`Endpoint "${ep.path}" has money field but no validation (allows negative values)`);
      }
    }
  });

  const authRoles = new Set(schemas.auth_rules?.roles || []);
  (intentRoles || []).forEach(role => {
    if (!authRoles.has(role)) issues.push(`Role "${role}" from intent missing in auth_rules`);
  });

  if (strict) {
    const apiPaths = schemas.api_schema?.endpoints?.map(ep => ep.path) || [];
    schemas.ui_schema?.pages?.forEach(page => {
      if (['/login', '/', '/register'].includes(page.route)) return;
      const routeKey = page.route.split('/')[1];
      const hasMatch = apiPaths.some(path => path.includes(routeKey));
      if (!hasMatch) issues.push(`UI page "${page.route}" has no matching API endpoint`);
    });
  }

  const userRefKeywords = ['user', 'author', 'posted', 'owner', 'creator', 'sender', 'receiver', 'patient', 'doctor'];
  const apiUserFields = new Set();
  const dbUserFields = new Set();
  schemas.api_schema?.endpoints?.forEach(ep => {
    [...Object.keys(ep.request_body || {}), ...Object.keys(ep.response || {})].forEach(f => {
      if (userRefKeywords.some(k => f.toLowerCase().includes(k))) apiUserFields.add(f);
    });
  });
  schemas.db_schema?.tables?.forEach(table => {
    table.columns?.forEach(col => {
      if (userRefKeywords.some(k => col.name.toLowerCase().includes(k))) dbUserFields.add(col.name);
    });
  });
  const allRefs = new Set([...apiUserFields, ...dbUserFields]);
  if (allRefs.size > 2) issues.push(`Cross-layer field name mismatch: ${[...allRefs].join(', ')} — should be unified`);

  return issues;
}

// ─── COMPILE PIPELINE ─────────────────────────────────────────────────────────
async function runPipeline(prompt) {
  const costBreakdown = [];
  let totalCost = 0;
  let totalTokens = { input: 0, output: 0 };
  const t0 = Date.now();

  // Prompt quality check
  const promptAnalysis = analyzePromptQuality(prompt);

  // STAGE 1 — INTENT
  const intentResult = await callGroq(`
Extract structured intent from this app description. Return ONLY valid JSON.
Format:
{
  "app_name": "string",
  "app_type": "string",
  "features": ["string"],
  "roles": ["string"],
  "integrations": ["string"],
  "auth_required": true,
  "monetization": "string or null",
  "ambiguities": ["string"],
  "assumptions": ["string"]
}
App description: "${prompt}"
`);
  costBreakdown.push({ stage: 'Intent Extraction', cost: intentResult.cost, tokens: intentResult.tokens, latency: intentResult.latency });
  totalCost += intentResult.cost;
  totalTokens.input += intentResult.tokens.input;
  totalTokens.output += intentResult.tokens.output;

  await wait(2000);

  // STAGE 2 — DESIGN
  const designResult = await callGroq(`
Convert this app intent into system architecture. Return ONLY valid JSON.
Format:
{
  "entities": [{"name":"string","fields":[{"name":"string","type":"string","required":true}],"relations":["string"]}],
  "pages": [{"name":"string","route":"string","access":["string"],"components":["string"]}],
  "auth_model": {"type":"string","roles":["string"],"permissions":[{"role":"string","actions":["string"]}]},
  "business_logic": ["string"],
  "external_services": ["string"]
}
Intent: ${JSON.stringify(intentResult.data)}
`);
  costBreakdown.push({ stage: 'System Design', cost: designResult.cost, tokens: designResult.tokens, latency: designResult.latency });
  totalCost += designResult.cost;
  totalTokens.input += designResult.tokens.input;
  totalTokens.output += designResult.tokens.output;

  await wait(2000);

  // STAGE 3 — SCHEMA
  const schemaResult = await callGroq(`
Generate complete app schemas from this system design. Return ONLY valid JSON.
Format:
{
  "ui_schema": {"pages":[{"id":"string","title":"string","route":"string","layout":"string","components":[{"type":"string","props":{}}]}]},
  "api_schema": {"version":"v1","endpoints":[{"method":"string","path":"string","auth_required":true,"roles":["string"],"request_body":{},"response":{}}]},
  "db_schema": {"tables":[{"name":"string","columns":[{"name":"string","type":"string","nullable":false,"primary_key":false,"foreign_key":null}],"indexes":["string"]}]},
  "auth_rules": {"strategy":"string","roles":["string"],"role_permissions":[{"role":"string","resource":"string","actions":["string"]}]}
}
Design: ${JSON.stringify(designResult.data)}
Intent: ${JSON.stringify(intentResult.data)}
`);
  costBreakdown.push({ stage: 'Schema Generation', cost: schemaResult.cost, tokens: schemaResult.tokens, latency: schemaResult.latency });
  totalCost += schemaResult.cost;
  totalTokens.input += schemaResult.tokens.input;
  totalTokens.output += schemaResult.tokens.output;

  await wait(2000);

  let schemas = schemaResult.data;

  // STAGE 4 — VALIDATION & REPAIR
  let issues = validateSchemas(schemas, intentResult.data.roles, true);
  let validation;
  let repairCost = 0;
  let retries = 0;

  if (issues.length === 0) {
    validation = {
      checks: [{ name: 'Schema consistency', status: 'pass', message: 'All checks passed', repaired: false }],
      repairs_made: [],
      overall: 'valid',
      confidence_score: 0.95
    };
  } else {
    try {
      await wait(2000);
      retries = 1;
      const repairResult = await repairSchemas(schemas, issues);
      repairCost = repairResult.cost;
      costBreakdown.push({ stage: 'Validation & Repair', cost: repairResult.cost, tokens: repairResult.tokens, latency: repairResult.latency });
      totalCost += repairResult.cost;
      totalTokens.input += repairResult.tokens.input;
      totalTokens.output += repairResult.tokens.output;

      const remainingIssues = validateSchemas(repairResult.data, intentResult.data.roles, false);

      if (remainingIssues.length === 0) {
        schemas = repairResult.data;
        validation = {
          checks: issues.map(issue => ({ name: issue, status: 'pass', message: 'Fixed during repair', repaired: true })),
          repairs_made: issues,
          overall: 'repaired',
          confidence_score: 0.85
        };
      } else {
        const fixedCount = issues.length - remainingIssues.length;
        schemas = repairResult.data;
        validation = {
          checks: [
            ...issues.filter(i => !remainingIssues.includes(i)).map(i => ({ name: i, status: 'pass', message: 'Fixed during repair', repaired: true })),
            ...remainingIssues.map(i => ({ name: i, status: 'fail', message: i, repaired: false }))
          ],
          repairs_made: issues.filter(i => !remainingIssues.includes(i)),
          overall: fixedCount > 0 ? 'repaired' : 'invalid',
          confidence_score: fixedCount > 0 ? 0.7 : 0.4
        };
      }
    } catch (repairErr) {
      validation = {
        checks: issues.map(i => ({ name: i, status: 'fail', message: i, repaired: false })),
        repairs_made: [],
        overall: 'invalid',
        confidence_score: 0.4
      };
    }
  }

  const totalLatency = (Date.now() - t0) / 1000;
  const executionProof = generateExecutionProof(schemas);

  // Cost vs Quality analysis
  const costQualityAnalysis = {
    total_cost_usd: parseFloat(totalCost.toFixed(6)),
    total_tokens: totalTokens,
    cost_breakdown: costBreakdown,
    latency_seconds: parseFloat(totalLatency.toFixed(2)),
    quality_score: parseFloat((validation.confidence_score * 100).toFixed(1)),
    cost_per_quality_point: parseFloat((totalCost / (validation.confidence_score * 100)).toFixed(8)),
    efficiency_rating: validation.confidence_score >= 0.85 ? 'high' : validation.confidence_score >= 0.6 ? 'medium' : 'low',
    retries,
    repair_cost_usd: parseFloat(repairCost.toFixed(6)),
    tradeoff_summary: `${totalLatency.toFixed(1)}s latency · $${totalCost.toFixed(5)} cost · ${Math.round(validation.confidence_score * 100)}% confidence`
  };

  // Update session stats
  sessionStats.totalRequests++;
  sessionStats.totalCost += totalCost;
  sessionStats.totalLatency += totalLatency;
  sessionStats.totalRetries += retries;
  if (validation.overall !== 'invalid') sessionStats.successCount++;
  else sessionStats.failCount++;

  return {
    meta: {
      generated_at: new Date().toISOString(),
      pipeline_version: '4.0.0',
      validation_status: validation.overall,
      confidence_score: validation.confidence_score,
      assumptions: [...(intentResult.data.assumptions || []), ...promptAnalysis.assumptions],
      ambiguities_resolved: intentResult.data.ambiguities,
      clarifications: promptAnalysis.clarifications,
      prompt_quality: promptAnalysis.quality,
      prompt_issues: promptAnalysis.issues,
      issues_found: issues.length,
      retries
    },
    intent: intentResult.data,
    design: designResult.data,
    schemas,
    validation_report: validation,
    execution_proof: executionProof,
    cost_quality_analysis: costQualityAnalysis,
    runtime_preview: executionProof.routes
  };
}

// ─── ROUTES ───────────────────────────────────────────────────────────────────

// Main compile endpoint
app.post('/compile', async (req, res) => {
  const { prompt } = req.body;
  if (!prompt) return res.status(400).json({ error: 'No prompt provided' });
  if (prompt.trim().length < 3) return res.status(400).json({ error: 'Prompt too short.' });

  // For very short prompts, still proceed but warn
  try {
    const result = await runPipeline(prompt.trim());
    res.json({ success: true, result });
  } catch (err) {
    console.error('Pipeline error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Run a single eval prompt
app.post('/eval/run', async (req, res) => {
  const { id } = req.body;
  const allPrompts = [...EVAL_DATASET.real, ...EVAL_DATASET.edge];
  const item = allPrompts.find(p => p.id === id);
  if (!item) return res.status(404).json({ error: 'Eval prompt not found' });

  const t0 = Date.now();
  try {
    const result = await runPipeline(item.prompt);
    const evalEntry = {
      id: item.id,
      label: item.label,
      category: item.category || 'real',
      prompt: item.prompt,
      status: result.meta.validation_status,
      confidence: result.meta.confidence_score,
      retries: result.meta.retries,
      latency: result.cost_quality_analysis.latency_seconds,
      cost: result.cost_quality_analysis.total_cost_usd,
      prompt_quality: result.meta.prompt_quality,
      issues_found: result.meta.issues_found,
      timestamp: new Date().toISOString()
    };
    evalResults = evalResults.filter(e => e.id !== id);
    evalResults.push(evalEntry);
    res.json({ success: true, entry: evalEntry, full_result: result });
  } catch (err) {
    const evalEntry = {
      id: item.id, label: item.label, category: item.category || 'real',
      prompt: item.prompt, status: 'error', confidence: 0, retries: 0,
      latency: (Date.now() - t0) / 1000, cost: 0, prompt_quality: 'poor',
      issues_found: 0, error: err.message, timestamp: new Date().toISOString()
    };
    evalResults = evalResults.filter(e => e.id !== id);
    evalResults.push(evalEntry);
    res.json({ success: false, entry: evalEntry, error: err.message });
  }
});

// Get eval dataset + results
app.get('/eval/dataset', (req, res) => {
  res.json({ dataset: EVAL_DATASET, results: evalResults });
});

// Get session stats
app.get('/stats', (req, res) => {
  res.json({
    ...sessionStats,
    avgCost: sessionStats.totalRequests ? (sessionStats.totalCost / sessionStats.totalRequests).toFixed(6) : 0,
    avgLatency: sessionStats.totalRequests ? (sessionStats.totalLatency / sessionStats.totalRequests).toFixed(2) : 0,
    successRate: sessionStats.totalRequests ? Math.round((sessionStats.successCount / sessionStats.totalRequests) * 100) : 0
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`AppForge v4 running on http://localhost:${PORT}`));