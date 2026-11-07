/** tiny DOM helpers (no framework: keeps the bundle small and the render loop free of reconciliation) */
export const $ = <T extends HTMLElement = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;
export const $$ = <T extends HTMLElement = HTMLElement>(sel: string, root: ParentNode = document) => Array.from(root.querySelectorAll(sel)) as T[];

export function setText(el: Element | null, v: string) {
  if (el && el.textContent !== v) el.textContent = v;
}

export const fmt = {
  m3: (v: number) => (Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(1)),
  pct: (v: number) => `${v.toFixed(0)}%`,
  cm: (v: number) => `${(v * 100).toFixed(1)} cm`,
  int: (v: number) => Math.round(v).toLocaleString('en-US'),
  s: (v: number) => `${v.toFixed(1)} s`,
};

export function toast(msg: string, kind: 'info' | 'warn' | 'ok' = 'info', ms = 2600) {
  const host = $('#toasts');
  if (!host) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add('in'));
  setTimeout(() => { el.classList.remove('in'); setTimeout(() => el.remove(), 300); }, ms);
}

export const icons = {
  explore: '<svg viewBox="0 0 24 24"><path d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm3.5 5.5-2 5-5 2 2-5 5-2Z"/></svg>',
  dig: '<svg viewBox="0 0 24 24"><path d="M4 19h16M6 15l3-3 3 3 2-2 4 4"/><path d="M14 4l4 4-3 3-4-4z"/></svg>',
  design: '<svg viewBox="0 0 24 24"><path d="M3 9h18M3 9l3 9h12l3-9"/><path d="M8 13h8" stroke-dasharray="2 2"/></svg>',
  scenario: '<svg viewBox="0 0 24 24"><path d="M12 3v2M5.6 5.6 7 7M3 12h2M17 7l1.4-1.4M19 12h2"/><path d="M8 16a4 4 0 1 1 8 0"/><path d="M4 20h16"/></svg>',
  auto: '<svg viewBox="0 0 24 24"><rect x="5" y="8" width="14" height="10" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M9 18v2M15 18v2"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/></svg>',
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  github: '<svg viewBox="0 0 24 24"><path d="M9 19c-4 1.3-4-2-6-2.5M15 21v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.3 4.3 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12 12 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.3 4.3 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/></svg>',
  undo: '<svg viewBox="0 0 24 24"><path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/></svg>',
  reset: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 4h3v16H7zM14 4h3v16h-3z"/></svg>',
  person: '<svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="2"/><path d="M12 8v7M8 11h8M10 21l2-6 2 6"/></svg>',
  close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  camera: '<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="12" rx="2"/><circle cx="12" cy="13" r="3.5"/><path d="M8 7l2-3h4l2 3"/></svg>',
  lidar: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="2"/><path d="M7 7a7 7 0 0 0 0 10M17 7a7 7 0 0 1 0 10M4 4a11 11 0 0 0 0 16M20 4a11 11 0 0 1 0 16"/></svg>',
};
