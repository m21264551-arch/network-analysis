import { mkdir, copyFile } from 'node:fs/promises';
await mkdir('dist/data', { recursive: true });
for (const file of ['index.html', 'script.js', 'styles.css', 'data/experiments.json']) await copyFile(file, `dist/${file}`);
