import sha1 from 'sha1';
import promptSync from 'prompt-sync';
const prompt = promptSync();

function getSha1() {
  var passphrase = prompt('Passphrase to use: ');
  console.log(`Sha1 is: ${sha1(passphrase)}`);
}

getSha1();
