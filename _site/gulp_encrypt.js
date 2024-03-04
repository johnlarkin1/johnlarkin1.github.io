import fs from 'fs';
import zlib from 'zlib';
import process from 'process';

import crypto from 'crypto-js';
import promptSync from 'prompt-sync';
const prompt = promptSync();

function encryptMarkdown() {
  process.chdir('./_posts');
  var fileIn = prompt('Markdown file to encrypt: ');
  var passphrase = prompt('Passphrase to use: ');
  var fileInSplit = fileIn.split('.');
  var fileOut = fileInSplit[0] + '-encrypted.' + fileInSplit.slice(1).join('.');
  try {
    fs.existsSync(fileIn);
  } catch (err) {
    console.error(`File does not exist: ${err}`);
  }
  const data = fs.readFileSync(fileIn, 'utf8');
  const delimiter = '---';
  const cutoff = 2;
  const tokens = data.split(delimiter).slice(cutoff);
  const result = tokens.join(delimiter);
  const encrypted = crypto.AES.encrypt(result, passphrase).toString();

  console.log('Original HTML: ', result);
  console.log('Replace Markdown content with: ');
  console.log(encrypted);
}

encryptMarkdown();
