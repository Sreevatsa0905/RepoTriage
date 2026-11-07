import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function runTests() {
  const fixturePath = path.join(__dirname, 'fixtures', 'analyze_response.json');
  const fullFixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

  const emptyFixture = {
    ...fullFixture,
    verified_claims: [],
    evidence_items: [],
    similar_issues: [],
    doc_excerpts: [],
    analysis: {
      issue_summary: 'Test empty edge case summary.',
      issue_type: 'question',
      likely_causes: [],
      recommended_next_steps: [],
      limitations: 'All lists empty for edge-case testing.'
    }
  };

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

  const pageErrors = [];
  page.on('pageerror', (err) => {
    pageErrors.push(err.message);
    console.error('PAGE ERROR:', err.message);
  });

  console.log('=== TEST 1: Full Fixture Render (Analyze Result View) ===');
  await page.route('**/api/analyze', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(fullFixture)
    });
  });

  await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' });
  await page.fill('textarea', 'branch failed as upstream configuration was not set');
  await page.click('button:has-text("Analyze Issue")');

  // Wait for results to render
  await page.waitForSelector('text=NLI Verified Claims Panel', { timeout: 10000 });
  await page.waitForSelector('text=Retrieved Evidence Items', { timeout: 10000 });
  await page.waitForSelector('text=Total Pipeline:', { timeout: 10000 });

  // Assertions
  const hasClaims = await page.isVisible('text=NLI Verified Claims Panel');
  const hasEvidence = await page.isVisible('text=Retrieved Evidence Items');
  const hasLatency = await page.isVisible('text=Total Pipeline:');
  const hasSummary = await page.isVisible('text=Issue Summary');
  const hasTrace = await page.isVisible('text=Trace:');
  console.log({ hasClaims, hasEvidence, hasLatency, hasSummary, hasTrace });

  if (!hasClaims || !hasEvidence || !hasLatency || !hasSummary || !hasTrace) {
    throw new Error('Test 1 Failed: Core result panels are missing!');
  }
  console.log('✓ Test 1 Passed: Claims panel, evidence cards, structured analysis, and latency footer rendered successfully!');

  // Capture real screenshot of completed analysis result for Step 8
  const screenshotPath = path.resolve(__dirname, '../../docs/screenshots/ui_result.png');
  await page.screenshot({ path: screenshotPath, fullPage: false });
  console.log(`✓ Screenshot of completed analysis result saved to: ${screenshotPath}`);

  console.log('\n=== TEST 2: Empty Claims & Evidence Edge Case ===');
  await page.unroute('**/api/analyze');
  await page.route('**/api/analyze', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(emptyFixture)
    });
  });

  await page.fill('textarea', 'testing empty edge case');
  await page.click('button:has-text("Analyze Issue")');

  await page.waitForSelector('text=No discrete factual claims', { timeout: 10000 });
  await page.waitForSelector('text=No evidence chunks retrieved', { timeout: 10000 });

  const hasEmptyClaimsMsg = await page.isVisible('text=No discrete factual claims');
  const hasEmptyEvidenceMsg = await page.isVisible('text=No evidence chunks retrieved');

  if (!hasEmptyClaimsMsg || !hasEmptyEvidenceMsg) {
    throw new Error('Test 2 Failed: Empty fallback messages not rendered!');
  }
  console.log('✓ Test 2 Passed: Empty claims and empty evidence handled gracefully without crashing!');

  console.log('\n=== TEST 3: API Error Handling (HTTP 429 Pipeline Busy) ===');
  await page.unroute('**/api/analyze');
  await page.route('**/api/analyze', async (route) => {
    await route.fulfill({
      status: 429,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Pipeline busy: another issue analysis is currently in progress. Please retry in a few moments.' })
    });
  });

  await page.click('button:has-text("Analyze Issue")');
  await page.waitForSelector('text=Pipeline busy', { timeout: 5000 });
  const hasBusyMsg = await page.isVisible('text=Pipeline busy');
  if (!hasBusyMsg) {
    throw new Error('Test 3 Failed: HTTP 429 error message not displayed!');
  }
  console.log('✓ Test 3 Passed: HTTP 429 pipeline busy error cleanly presented in alert UI!');

  console.log('\n=== TEST 4: Search-Only and Evaluation Tab Health ===');
  await page.click('text=Evaluation & Benchmarks');
  await page.waitForSelector('text=Retrieval 4-Way Ablation', { timeout: 5000 });
  await page.waitForSelector('text=Guardrail Benchmark B1', { timeout: 5000 });
  await page.waitForSelector('text=Guardrail Benchmark B2', { timeout: 5000 });
  await page.waitForSelector('text=Automated Perturbation Test', { timeout: 5000 });

  const hasAblation = await page.isVisible('text=Retrieval 4-Way Ablation');
  const hasB1 = await page.isVisible('text=Guardrail Benchmark B1');
  if (!hasAblation || !hasB1) {
    throw new Error('Test 4 Failed: Evaluation tab content missing!');
  }
  console.log('✓ Test 4 Passed: Evaluation & Benchmarks tab verified and fully functional!');

  if (pageErrors.length > 0) {
    throw new Error(`Encountered ${pageErrors.length} uncaught page errors during test suite: ${pageErrors.join('; ')}`);
  }

  console.log('\n======================================================');
  console.log('ALL FRONTEND SUITE TESTS PASSED (0 UNCAUGHT EXCEPTIONS)');
  console.log('======================================================');

  await browser.close();
}

runTests().catch((err) => {
  console.error('FRONTEND TEST SUITE FAILED:', err);
  process.exit(1);
});
