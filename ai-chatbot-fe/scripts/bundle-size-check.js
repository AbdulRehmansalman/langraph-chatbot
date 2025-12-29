#!/usr/bin/env node

/**
 * Bundle Size Monitoring Script
 * Checks if bundle sizes exceed defined thresholds
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distPath = path.join(__dirname, '../dist');

// Bundle size thresholds (in KB)
const THRESHOLDS = {
  main: 300, // Main bundle threshold
  vendor: 200, // Vendor bundle threshold
  chunk: 100, // Individual chunk threshold
  total: 500, // Total bundle size threshold
};

function formatSize(bytes) {
  return (bytes / 1024).toFixed(2) + ' KB';
}

function checkBundleSize() {
  if (!fs.existsSync(distPath)) {
    console.error('❌ Build directory not found. Run "npm run build" first.');
    process.exit(1);
  }

  const assetPath = path.join(distPath, 'assets');

  if (!fs.existsSync(assetPath)) {
    console.error('❌ Assets directory not found in build output.');
    process.exit(1);
  }

  const files = fs.readdirSync(assetPath);
  const jsFiles = files.filter(file => file.endsWith('.js'));
  const cssFiles = files.filter(file => file.endsWith('.css'));

  let totalSize = 0;
  let violations = [];

  console.log('📦 Bundle Size Analysis\n');
  console.log('JavaScript Files:');
  console.log('─'.repeat(50));

  jsFiles.forEach(file => {
    const filePath = path.join(assetPath, file);
    const stats = fs.statSync(filePath);
    const sizeKB = stats.size / 1024;
    totalSize += stats.size;

    const status = sizeKB > THRESHOLDS.chunk ? '❌' : '✅';
    console.log(`${status} ${file}: ${formatSize(stats.size)}`);

    // Check specific bundle types
    if (file.includes('vendor') && sizeKB > THRESHOLDS.vendor) {
      violations.push(`Vendor bundle ${file} exceeds ${THRESHOLDS.vendor}KB threshold`);
    } else if (file.includes('index') && sizeKB > THRESHOLDS.main) {
      violations.push(`Main bundle ${file} exceeds ${THRESHOLDS.main}KB threshold`);
    } else if (sizeKB > THRESHOLDS.chunk) {
      violations.push(`Chunk ${file} exceeds ${THRESHOLDS.chunk}KB threshold`);
    }
  });

  console.log('\nCSS Files:');
  console.log('─'.repeat(50));

  cssFiles.forEach(file => {
    const filePath = path.join(assetPath, file);
    const stats = fs.statSync(filePath);
    totalSize += stats.size;
    console.log(`✅ ${file}: ${formatSize(stats.size)}`);
  });

  const totalSizeKB = totalSize / 1024;
  console.log('\nSummary:');
  console.log('─'.repeat(50));
  console.log(`Total Bundle Size: ${formatSize(totalSize)}`);
  console.log(`Threshold: ${THRESHOLDS.total}KB`);

  if (totalSizeKB > THRESHOLDS.total) {
    violations.push(
      `Total bundle size ${formatSize(totalSize)} exceeds ${THRESHOLDS.total}KB threshold`
    );
  }

  if (violations.length > 0) {
    console.log('\n❌ Bundle Size Violations:');
    violations.forEach(violation => {
      console.log(`  • ${violation}`);
    });
    console.log('\n💡 Consider:');
    console.log('  • Code splitting with dynamic imports');
    console.log('  • Tree shaking unused dependencies');
    console.log('  • Lazy loading components');
    console.log('  • Removing unused packages');
    process.exit(1);
  } else {
    console.log('\n✅ All bundle sizes are within acceptable limits!');

    // Performance recommendations
    if (totalSizeKB > THRESHOLDS.total * 0.8) {
      console.log('\n⚠️  Bundle size is approaching the threshold. Consider optimizations.');
    }
  }
}

// Gzip size estimation (rough approximation)
function estimateGzipSize(files) {
  console.log('\n📊 Estimated Gzip Sizes:');
  console.log('─'.repeat(50));

  files.forEach(file => {
    const filePath = path.join(distPath, 'assets', file);
    const stats = fs.statSync(filePath);
    // Rough gzip estimation: ~30% of original size for JS, ~20% for CSS
    const estimatedGzip = file.endsWith('.js') ? stats.size * 0.3 : stats.size * 0.2;
    console.log(`${file}: ${formatSize(estimatedGzip)} (estimated)`);
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  checkBundleSize();

  // Show estimated gzip sizes
  const assetPath = path.join(distPath, 'assets');
  if (fs.existsSync(assetPath)) {
    const files = fs
      .readdirSync(assetPath)
      .filter(file => file.endsWith('.js') || file.endsWith('.css'));
    estimateGzipSize(files);
  }
}

export { checkBundleSize };
