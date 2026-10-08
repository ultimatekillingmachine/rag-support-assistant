// Chat page logic: talks to POST /ask/stream and renders server-sent events.
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('question');
const tokenInput = document.getElementById('token');
const history = [];  // { role, content } — sent back with each turn

const EXAMPLES = [
  'Сколько идёт доставка в Москву?',
  'Сколько дней заказ хранится в пункте выдачи?',
  'Какая гарантия на холодильники?',
  'А если товар бракованный?',
  'Есть ли доставка дронами?',
];

document.getElementById('hints').innerHTML = EXAMPLES
  .map((text) => `<button type="button" data-q="${text}">${text}</button>`).join('');
document.getElementById('hints').addEventListener('click', (event) => {
  if (event.target.dataset.q) { input.value = event.target.dataset.q; form.requestSubmit(); }
});

function bubble(kind, text = '') {
  const node = document.createElement('div');
  node.className = `msg ${kind}`;
  node.textContent = text;
  log.appendChild(node);
  node.scrollIntoView({ block: 'end' });
  return node;
}

function renderSources(node, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement('div');
  box.className = 'sources';
  box.innerHTML = 'Источники:<ol>' + sources
    .map((s) => `<li>${s.title} <code>${s.slug}</code> (обновлено ${s.updated_at || '—'})</li>`)
    .join('') + '</ol>';
  node.appendChild(box);
}

function appendToken(node, text, accumulated) {
  const textNode = node.firstChild && node.firstChild.nodeType === Node.TEXT_NODE
    ? node.firstChild
    : node.insertBefore(document.createTextNode(''), node.firstChild);
  textNode.nodeValue = accumulated + text;
  return accumulated + text;
}

async function ask(payload, answerNode) {
  const response = await fetch('/ask/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${tokenInput.value}`,
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    answerNode.textContent = `Ошибка ${response.status}: проверьте токен доступа`;
    answerNode.classList.add('refused');
    return null;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answer = '';
  let refused = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split('\n\n');
    buffer = frames.pop();
    for (const frame of frames) {
      if (!frame.startsWith('data:')) continue;
      const event = JSON.parse(frame.slice(5).trim());
      if (event.type === 'sources') {
        renderSources(answerNode, event.sources);
      } else if (event.type === 'token') {
        answer = appendToken(answerNode, event.text, answer);
      } else if (event.type === 'done') {
        refused = Boolean(event.refused);
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = `${event.latency_ms} мс` +
          (payload.history.length ? ' · с учётом диалога' : '');
        answerNode.appendChild(meta);
      } else if (event.type === 'error') {
        answerNode.textContent = event.message;
        answerNode.classList.add('refused');
      }
    }
  }
  if (refused) answerNode.classList.add('refused');
  return answer;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  bubble('user', question);

  const answerNode = bubble('bot', '…');
  const payload = { question, history: history.slice(-8) };
  history.push({ role: 'user', content: question });

  try {
    const answer = await ask(payload, answerNode);
    if (answer) history.push({ role: 'assistant', content: answer });
  } catch (error) {
    answerNode.textContent = 'Не удалось связаться с сервисом: ' + error;
    answerNode.classList.add('refused');
  }
});
