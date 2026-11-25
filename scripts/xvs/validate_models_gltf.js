const fs = require('fs');
const path = require('path');
const validator = require('../../tools/gltf_validator/node_modules/gltf-validator');

async function validateGlb(filePath) {
  const data = fs.readFileSync(filePath);
  const result = await validator.validateBytes(new Uint8Array(data));
  const errors = result.issues.messages.filter(m => m.severity === 0);
  const warnings = result.issues.messages.filter(m => m.severity === 1);
  return {
    filePath,
    numErrors: result.issues.numErrors,
    numWarnings: result.issues.numWarnings,
    numInfos: result.issues.numInfos,
    errors,
    warnings
  };
}

async function main() {
  const assetsDir = path.resolve(__dirname, '../../assets/traffic');
  const files = [];

  function scanDir(dir) {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        scanDir(full);
      } else if (ent.isFile() && ent.name.endsWith('.glb')) {
        files.push(full);
      }
    }
  }

  scanDir(assetsDir);
  console.log(`Found ${files.length} GLB files to validate...`);

  let totalErrors = 0;
  for (const f of files) {
    const rel = path.relative(assetsDir, f).replace(/\\/g, '/');
    const res = await validateGlb(f);
    totalErrors += res.numErrors;
    if (res.numErrors > 0) {
      console.error(`[FAIL] ${rel}: ${res.numErrors} errors, ${res.numWarnings} warnings`);
      for (const m of res.errors) {
        console.error(`   Error [${m.code}]: ${m.message}`);
      }
    } else {
      console.log(`[PASS] ${rel}: 0 errors, ${res.numWarnings} warnings`);
    }
  }

  console.log(`\nValidation complete: ${totalErrors} total errors.`);
  if (totalErrors > 0) {
    process.exit(1);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
