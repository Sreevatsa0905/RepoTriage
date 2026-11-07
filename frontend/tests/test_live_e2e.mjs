import { chromium } from 'playwright';

async function testLiveE2E() {
  console.log('Starting Live End-to-End Test (No Mocks)...');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

  const pageErrors = [];
  page.on('pageerror', (err) => {
    pageErrors.push(err.message);
    console.error('LIVE UNCAUGHT PAGE ERROR:', err.message);
  });

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      console.error('LIVE CONSOLE ERROR:', msg.text());
    }
  });

  console.log('Navigating to http://127.0.0.1:5173 ...');
  await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' });

  console.log('Selecting repository pydantic/pydantic...');
  await page.selectOption('select', 'pydantic/pydantic');

  console.log('Filling issue text...');
  await page.fill('textarea', 'branch failed as upstream configuration was not set');

  console.log('Clicking "Analyze Issue" and waiting for real pipeline execution (~25-35s)...');
  await page.click('button:has-text("Analyze Issue")');

  // Wait for the real API response and render
  await page.waitForResponse(
    (resp) => resp.url().includes('/api/analyze') && resp.status() === 200,
    { timeout: 90000 }
  );
  console.log('Live /api/analyze response received! Checking render...');

  // Wait for claims panel to appear
  await page.waitForSelector('text=NLI Verified Claims Panel', { timeout: 15000 });
  await page.waitForSelector('text=Retrieved Evidence Items', { timeout: 15000 });
  await page.waitForSelector('text=Total Pipeline:', { timeout: 15000 });

  const rootContent = await page.innerHTML('#root');
  console.log('Root HTML length:', rootContent.length);

  if (rootContent.length === 0) {
    throw new Error('FAIL: Root div is empty!');
  }

  // Capture real live screenshot
  await page.screenshot({ path: 'docs/screenshots/ui_result.png', fullPage: false });
  console.log('✓ Successfully captured real live screenshot to docs/screenshots/ui_result.png!');

  if (pageErrors.length > 0) {
    throw new Error(`Encountered page errors: ${pageErrors.join('; ')}`);
  }

  console.log('✓ Live End-to-End Test PASSED completely with 0 errors!');
  await browser.close();
}

testLiveE2E().catch((err) => {
  console.error('Live E2E Test FAILED:', err);
  process.exit(1);
});
