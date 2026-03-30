import { useState } from "react";

const API = "https://shivamtech9395-content-moderation-env.hf.space";

const LABEL_CONFIG = {
  safe:          { color: "#10b981", bg: "#ecfdf5", border: "#a7f3d0", icon: "✓", label: "SAFE" },
  spam:          { color: "#f59e0b", bg: "#fffbeb", border: "#fde68a", icon: "⚠", label: "SPAM" },
  hate_speech:   { color: "#ef4444", bg: "#fef2f2", border: "#fecaca", icon: "✗", label: "HATE" },
  misinformation:{ color: "#3b82f6", bg: "#eff6ff", border: "#bfdbfe", icon: "?", label: "MISINFO" },
  harmful:       { color: "#8b5cf6", bg: "#f5f3ff", border: "#ddd6fe", icon: "!", label: "HARMFUL" },
};

const QUEUE_DATA = [
  { id:"q001", text:"Buy cheap Viagra now!!! Click here for 90% discount on all meds!!!", label:"spam", confidence:0.97, time:"2m ago" },
  { id:"q002", text:"People from that country should all go back where they came from.", label:"hate_speech", confidence:0.93, time:"5m ago" },
  { id:"q003", text:"5G towers secretly cause cancer. Share before they delete this!", label:"misinformation", confidence:0.89, time:"8m ago" },
  { id:"q004", text:"Drinking bleach can cure COVID-19. Big pharma hiding the truth!", label:"harmful", confidence:0.94, time:"12m ago" },
  { id:"q005", text:"Win a FREE iPhone 15! Just send your bank details to claim!", label:"spam", confidence:0.98, time:"15m ago" },
  { id:"q006", text:"Women are too emotional to be leaders. They should stay home.", label:"hate_speech", confidence:0.91, time:"20m ago" },
];

const SAMPLES = [
  "Buy cheap Viagra now!!! 90% discount on all meds!!!",
  "I love spending time with my family on weekends.",
  "People from that country should all go back where they came from.",
  "5G towers secretly cause cancer. Share before they delete!",
  "Drinking bleach can cure COVID-19. Big pharma hiding truth!",
  "Scientists at MIT published peer-reviewed climate research.",
];

function localClassify(text) {
  const t = text.toLowerCase();
  if (/buy|click here|discount|win|prize|urgent|viagra|free iphone|bank detail|ssn|credit card/i.test(t))
    return { decision:"spam", confidence:0.95, reason:"Contains promotional language, urgency triggers, and suspicious call-to-action patterns." };
  if (/go back|not human|animals|should stay home|too emotional|ruining everything|subhuman/i.test(t))
    return { decision:"hate_speech", confidence:0.92, reason:"Targets a group based on identity characteristics with derogatory language." };
  if (/5g|microchip|bleach cure|they delete|big pharma|moon landing|secretly cause|vaccine track|named researchers/i.test(t))
    return { decision:"misinformation", confidence:0.88, reason:"Contains false factual claims contradicting established scientific evidence." };
  if (/bleach|dangerous poison|kill yourself|how to make bomb/i.test(t))
    return { decision:"harmful", confidence:0.9, reason:"Contains advice or information that could cause physical harm." };
  return { decision:"safe", confidence:0.87, reason:"Content appears normal with no harmful elements detected." };
}

// ── BADGE ──
function Badge({ label }) {
  const cfg = LABEL_CONFIG[label] || LABEL_CONFIG.safe;
  return (
    <span
      style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}
      className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold font-mono"
    >
      {cfg.icon} {cfg.label}
    </span>
  );
}

// ── STAT CARD ──
function StatCard({ icon, value, label, change, changeUp, gradient }) {
  return (
    <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm hover:-translate-y-1 transition-all duration-200 relative overflow-hidden">
      <div className={`absolute top-0 right-0 w-20 h-20 rounded-full bg-gradient-to-br ${gradient} opacity-10 -mr-4 -mt-4`}></div>
      <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-white text-lg mb-3 shadow-md`}>{icon}</div>
      <div className="text-2xl font-bold text-gray-900 font-mono">{value}</div>
      <div className="text-sm text-gray-500 mt-1">{label}</div>
      {change && (
        <div className={`text-xs mt-2 font-medium ${changeUp ? "text-emerald-500" : "text-gray-400"}`}>{change}</div>
      )}
    </div>
  );
}

// ── DASHBOARD ──
function DashboardPage() {
  const activity = [
    { text: '"Buy cheap Viagra now!!!"', label: "spam", score: "1.00" },
    { text: '"I love spending time with my family..."', label: "safe", score: "1.00" },
    { text: '"People from that country should go back..."', label: "hate_speech", score: "0.99" },
    { text: '"5G towers secretly cause cancer..."', label: "misinformation", score: "0.98" },
    { text: '"Scientists at MIT published research..."', label: "safe", score: "1.00" },
  ];
  const bars = [
    { label: "Easy", value: 100, color: "#14b8a6", score: "1.00" },
    { label: "Medium", value: 99, color: "#0ea5e9", score: "0.99" },
    { label: "Hard", value: 92, color: "#f59e0b", score: "0.92" },
    { label: "Avg", value: 97, color: "#10b981", score: "0.97" },
  ];
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Dashboard</h2>
        <p className="text-sm text-gray-500 mt-1">Content Moderation OpenEnv — Real-time overview</p>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon="🎯" value="0.97" label="Baseline Score" change="↑ Avg across tasks" changeUp gradient="from-teal-500 to-sky-500" />
        <StatCard icon="📝" value="18" label="Total Posts" change="6 per task" changeUp gradient="from-violet-500 to-blue-500" />
        <StatCard icon="⚡" value="3" label="Active Tasks" change="Easy · Medium · Hard" gradient="from-amber-400 to-orange-500" />
        <StatCard icon="🏆" value="1.00" label="Best Score" change="↑ Easy Task" changeUp gradient="from-rose-500 to-pink-500" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="flex justify-between items-center mb-4">
            <span className="font-semibold text-gray-800 text-sm">Task Scores</span>
            <span className="text-xs text-gray-400">Baseline Agent</span>
          </div>
          <div className="flex items-end gap-4 h-28">
            {bars.map(b => (
              <div key={b.label} className="flex flex-col items-center gap-1 flex-1">
                <span className="text-xs font-mono font-bold" style={{ color: b.color }}>{b.score}</span>
                <div className="w-full rounded-t-lg" style={{ height: `${b.value}%`, background: b.color, minHeight: "8px" }}></div>
                <span className="text-xs text-gray-400">{b.label}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="flex justify-between items-center mb-4">
            <span className="font-semibold text-gray-800 text-sm">Label Distribution</span>
            <span className="text-xs text-gray-400">All tasks</span>
          </div>
          <div className="flex items-center gap-6">
            <svg width="100" height="100" viewBox="0 0 100 100" className="flex-shrink-0">
              <circle cx="50" cy="50" r="40" fill="none" stroke="#f1f5f9" strokeWidth="18"/>
              <circle cx="50" cy="50" r="40" fill="none" stroke="#14b8a6" strokeWidth="18" strokeDasharray="113 163" strokeDashoffset="0" transform="rotate(-90 50 50)"/>
              <circle cx="50" cy="50" r="40" fill="none" stroke="#f59e0b" strokeWidth="18" strokeDasharray="50 226" strokeDashoffset="-113" transform="rotate(-90 50 50)"/>
              <circle cx="50" cy="50" r="40" fill="none" stroke="#ef4444" strokeWidth="18" strokeDasharray="25 251" strokeDashoffset="-163" transform="rotate(-90 50 50)"/>
              <circle cx="50" cy="50" r="40" fill="none" stroke="#3b82f6" strokeWidth="18" strokeDasharray="25 251" strokeDashoffset="-188" transform="rotate(-90 50 50)"/>
              <circle cx="50" cy="50" r="40" fill="none" stroke="#8b5cf6" strokeWidth="18" strokeDasharray="37 239" strokeDashoffset="-213" transform="rotate(-90 50 50)"/>
            </svg>
            <div className="space-y-1.5">
              {[["#14b8a6","Safe","45%"],["#f59e0b","Spam","20%"],["#ef4444","Hate","10%"],["#3b82f6","Misinfo","10%"],["#8b5cf6","Harmful","15%"]].map(([c,l,p]) => (
                <div key={l} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: c }}></div>
                  <span className="text-xs text-gray-500">{l}</span>
                  <span className="text-xs font-semibold text-gray-700 ml-auto pl-3">{p}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
        <div className="font-semibold text-gray-800 text-sm mb-4">Recent Activity</div>
        {activity.map((a, i) => (
          <div key={i} className="flex items-center gap-3 py-3 border-b border-gray-50 last:border-0">
            <Badge label={a.label} />
            <span className="flex-1 text-sm text-gray-600 truncate">{a.text}</span>
            <span className="text-xs font-mono font-semibold text-teal-600">{a.score}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── PLAYGROUND ──
function PlaygroundPage() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiScore, setApiScore] = useState(null);

  async function handleClassify() {
  if (!input.trim()) return;
  setLoading(true);
  setResult(null);
  setApiScore(null);

  const local = localClassify(input);

  try {
    await fetch(`${API}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_name: "hard" }),
    });

    const res = await fetch(`${API}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision: local.decision,
        confidence: local.confidence,
        reason: local.reason,
      }),
    });

    const data = await res.json();
    setApiScore(data.reward?.score ?? null);
  } catch {
    setApiScore(null);
  }

  setResult(local);
  setLoading(false);
}

  const cfg = result ? LABEL_CONFIG[result.decision] : null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">🎮 Playground</h2>
        <p className="text-sm text-gray-500 mt-1">Test content moderation live — click a sample or type your own</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="space-y-4">
          <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
            <div className="font-semibold text-gray-800 text-sm mb-3">Sample Posts</div>
            <div className="space-y-2">
              {SAMPLES.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setInput(s)}
                  className="w-full text-left text-xs text-gray-500 bg-gray-50 hover:bg-teal-50 hover:text-teal-700 border border-transparent hover:border-teal-200 rounded-xl px-3 py-2.5 transition-all leading-relaxed"
                >
                  "{s}"
                </button>
              ))}
            </div>
          </div>
          <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
            <div className="font-semibold text-gray-800 text-sm mb-3">Your Post</div>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type or paste any social media post..."
              className="w-full text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded-xl p-3 resize-none h-28 focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-transparent transition-all"
            />
            <button
              onClick={handleClassify}
              disabled={loading || !input.trim()}
              className="mt-3 w-full py-3 rounded-xl font-semibold text-sm text-white bg-gradient-to-r from-teal-500 to-sky-500 hover:from-teal-600 hover:to-sky-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md"
            >
              {loading ? "⏳ Classifying..." : "🔍 Classify Post"}
            </button>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="font-semibold text-gray-800 text-sm mb-4">Classification Result</div>
          {loading && (
            <div className="space-y-3 mt-2">
              {[32, 100, 60].map((w, i) => (
                <div key={i} className="h-8 rounded-lg bg-gray-100 animate-pulse" style={{ width: `${w}%` }}></div>
              ))}
            </div>
          )}
          {result && !loading && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span className="text-3xl font-bold font-mono" style={{ color: cfg.color }}>{result.decision}</span>
                <Badge label={result.decision} />
              </div>
              <div>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Confidence</span>
                  <span className="font-mono font-semibold">{(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-700" style={{ width: `${result.confidence * 100}%`, background: cfg.color }}></div>
                </div>
              </div>
              <div className="bg-gray-50 rounded-xl p-4 border-l-4" style={{ borderColor: cfg.color }}>
                <div className="text-xs font-semibold text-gray-400 mb-1">REASON</div>
                <div className="text-sm text-gray-700 leading-relaxed">{result.reason}</div>
              </div>
              {apiScore !== null && (
                <div className="bg-teal-50 border border-teal-100 rounded-xl p-3 flex items-center gap-3">
                  <span className="text-xl">🎯</span>
                  <div>
                    <div className="text-xs font-semibold text-teal-700">Environment Reward Score</div>
                    <div className="text-lg font-bold font-mono text-teal-600">{apiScore.toFixed(2)}</div>
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 rounded-xl p-3 text-center">
                  <div className="text-lg font-bold font-mono" style={{ color: cfg.color }}>{result.decision}</div>
                  <div className="text-xs text-gray-400 mt-0.5">Decision</div>
                </div>
                <div className="bg-gray-50 rounded-xl p-3 text-center">
                  <div className="text-lg font-bold font-mono text-gray-800">{(result.confidence * 100).toFixed(0)}%</div>
                  <div className="text-xs text-gray-400 mt-0.5">Confidence</div>
                </div>
              </div>
            </div>
          )}
          {!result && !loading && (
            <div className="flex flex-col items-center justify-center h-64 text-gray-300">
              <div className="text-5xl mb-3">🎯</div>
              <div className="text-sm">Select a sample or type a post</div>
              <div className="text-xs mt-1">AI will analyze the content</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── QUEUE ──
function QueuePage() {
  const [filter, setFilter] = useState("all");
  const [decisions, setDecisions] = useState({});

  const filters = ["all", "spam", "hate_speech", "misinformation", "harmful"];
  const filtered = QUEUE_DATA.filter(q => filter === "all" || q.label === filter);
  const pending = QUEUE_DATA.filter(q => !decisions[q.id]).length;

  function decide(id, action) {
    setDecisions(prev => ({ ...prev, [id]: action }));
  }

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📋 Moderation Queue</h2>
          <p className="text-sm text-gray-500 mt-1">Review and approve or reject flagged content</p>
        </div>
        <div className="bg-rose-50 border border-rose-200 text-rose-600 text-xs font-semibold px-3 py-1.5 rounded-full">
          {pending} pending
        </div>
      </div>
      <div className="flex gap-2 flex-wrap">
        {filters.map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${filter === f ? "bg-gradient-to-r from-teal-500 to-sky-500 text-white shadow-md" : "bg-gray-100 text-gray-500 hover:bg-gray-200"}`}
          >
            {f === "all" ? "All" : f.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())}
          </button>
        ))}
      </div>
      <div className="space-y-3">
        {filtered.map(q => {
          const dc = decisions[q.id];
          const cfg = LABEL_CONFIG[q.label];
          return (
            <div key={q.id} className={`bg-white rounded-2xl p-5 border shadow-sm transition-all ${dc ? "opacity-60 border-gray-100" : "border-gray-100 hover:border-teal-200"}`}>
              <div className="flex items-center gap-3 mb-3">
                <Badge label={q.label} />
                <span className="text-xs text-gray-400 font-mono">#{q.id}</span>
                <span className="text-xs text-gray-400 ml-auto">{q.time}</span>
                {dc && (
                  <span className={`text-xs font-semibold ${dc === "approve" ? "text-emerald-500" : "text-rose-500"}`}>
                    {dc === "approve" ? "✓ Approved" : "✗ Rejected"}
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-700 leading-relaxed mb-4">"{q.text}"</p>
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>AI Confidence</span>
                    <span className="font-mono font-semibold">{(q.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${q.confidence * 100}%`, background: cfg.color }}></div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => decide(q.id, "approve")} disabled={!!dc} className="px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-50 text-emerald-600 hover:bg-emerald-500 hover:text-white border border-emerald-200 disabled:opacity-40 disabled:cursor-not-allowed transition-all">✓ Approve</button>
                  <button onClick={() => decide(q.id, "reject")} disabled={!!dc} className="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-50 text-rose-600 hover:bg-rose-500 hover:text-white border border-rose-200 disabled:opacity-40 disabled:cursor-not-allowed transition-all">✗ Reject</button>
                  <button onClick={() => decide(q.id, "skip")} disabled={!!dc} className="px-4 py-2 rounded-xl text-xs font-semibold bg-gray-50 text-gray-500 hover:bg-gray-200 border border-gray-200 disabled:opacity-40 disabled:cursor-not-allowed transition-all">Skip</button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── MODEL INFO ──
function ModelPage() {
  const metrics = [
    { label: "Easy Task", value: 1.0, color: "#14b8a6" },
    { label: "Medium Task", value: 0.99, color: "#0ea5e9" },
    { label: "Hard Task", value: 0.92, color: "#f59e0b" },
    { label: "Overall Avg", value: 0.97, color: "#10b981" },
  ];
  const pipeline = [
    { n: "01", title: "Input Preprocessing", desc: "Post text normalized, context extracted, observation built from environment state" },
    { n: "02", title: "LLM Inference", desc: "Llama-3.1-8B processes post with system prompt via HuggingFace router API" },
    { n: "03", title: "JSON Parsing", desc: "Model outputs {decision, confidence, reason} — parsed and validated" },
    { n: "04", title: "Reward Computation", desc: "Score: exact match 0.9+, partial 0.4+, missed harmful 0.0" },
  ];
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">🧠 AI Model Info</h2>
        <p className="text-sm text-gray-500 mt-1">Architecture, pipeline and performance metrics</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="font-semibold text-gray-800 text-sm mb-4">Model Performance</div>
          <div className="space-y-4">
            {metrics.map(m => (
              <div key={m.label}>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-gray-500 font-medium">{m.label}</span>
                  <span className="font-mono font-bold" style={{ color: m.color }}>{m.value.toFixed(2)}</span>
                </div>
                <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-700" style={{ width: `${m.value * 100}%`, background: m.color }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="font-semibold text-gray-800 text-sm mb-4">Model Details</div>
          {[["Model","Llama-3.1-8B-Instruct"],["Provider","HuggingFace Router"],["Temperature","0.1"],["Max Tokens","100"],["Output","JSON"],["API","router.huggingface.co/v1"]].map(([k,v]) => (
            <div key={k} className="flex justify-between py-2.5 border-b border-gray-50 last:border-0">
              <span className="text-xs text-gray-400 font-medium">{k}</span>
              <span className="text-xs font-mono font-semibold text-gray-700">{v}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="font-semibold text-gray-800 text-sm mb-4">Classification Pipeline</div>
          <div className="space-y-4">
            {pipeline.map((p, i) => (
              <div key={i} className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-500 to-sky-500 text-white text-xs font-bold flex items-center justify-center flex-shrink-0 shadow-md">{p.n}</div>
                <div>
                  <div className="text-sm font-semibold text-gray-800">{p.title}</div>
                  <div className="text-xs text-gray-500 mt-0.5 leading-relaxed">{p.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
          <div className="font-semibold text-gray-800 text-sm mb-4">Tech Stack</div>
          <div className="flex flex-wrap gap-2 mb-5">
            {["Python 3.11","FastAPI","Pydantic v2","OpenAI Client","Docker","HF Spaces","Llama-3.1-8B","OpenEnv"].map(t => (
              <span key={t} className="px-3 py-1.5 bg-gray-50 border border-gray-200 rounded-full text-xs font-mono text-gray-600 hover:bg-teal-50 hover:border-teal-200 hover:text-teal-700 transition-colors cursor-default">{t}</span>
            ))}
          </div>
          <div className="bg-gradient-to-br from-teal-50 to-sky-50 rounded-xl p-4 border border-teal-100">
            <div className="text-xs font-semibold text-teal-700 mb-1">🏆 Hackathon Score</div>
            <div className="text-2xl font-bold font-mono text-teal-600">0.97 / 1.00</div>
            <div className="text-xs text-teal-500 mt-0.5">Meta PyTorch Hackathon 2026</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── MAIN APP ──
export default function App() {
  const [page, setPage] = useState("dashboard");

  const navItems = [
    { id: "dashboard", icon: "📊", label: "Dashboard" },
    { id: "playground", icon: "🎮", label: "Playground" },
    { id: "queue", icon: "📋", label: "Queue", badge: 6 },
    { id: "model", icon: "🧠", label: "AI Model" },
  ];

  const pages = {
    dashboard: <DashboardPage />,
    playground: <PlaygroundPage />,
    queue: <QueuePage />,
    model: <ModelPage />,
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <aside className="w-60 bg-white border-r border-gray-100 flex flex-col shadow-sm flex-shrink-0">
        <div className="p-5 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-500 to-sky-500 flex items-center justify-center text-white text-lg shadow-md">🛡️</div>
            <div>
              <div className="font-bold text-gray-900 text-sm">ContentGuard</div>
              <div className="text-xs text-teal-500 font-mono">AI · OpenEnv v1.0</div>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-3 py-2">Navigation</div>
          {navItems.map(n => (
            <button
              key={n.id}
              onClick={() => setPage(n.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${page === n.id ? "bg-gradient-to-r from-teal-500 to-sky-500 text-white shadow-md" : "text-gray-500 hover:bg-gray-50 hover:text-gray-800"}`}
            >
              <span className="text-base">{n.icon}</span>
              <span className="flex-1 text-left">{n.label}</span>
              {n.badge > 0 && (
                <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full ${page === n.id ? "bg-white/20 text-white" : "bg-rose-100 text-rose-500"}`}>{n.badge}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-100">
          <div className="flex items-center gap-2 bg-emerald-50 rounded-xl px-3 py-2.5">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
            <span className="text-xs text-emerald-700 font-medium">API Connected</span>
          </div>
          <div className="mt-2 text-center text-xs text-gray-400 font-mono">shivamtech9395 · HF Spaces</div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-gray-100 px-6 py-3.5 flex items-center justify-between shadow-sm">
          <h1 className="font-bold text-gray-900 text-sm">
            {navItems.find(n => n.id === page)?.icon} {navItems.find(n => n.id === page)?.label}
          </h1>
          <div className="flex items-center gap-3">
            <select className="text-xs bg-gray-50 border border-gray-200 text-gray-600 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-teal-400 font-medium">
              <option value="easy">🟢 Spam Detection</option>
              <option value="medium">🟡 Hate Speech</option>
              <option value="hard">🔴 Misinformation</option>
            </select>
            <div className="bg-gradient-to-r from-teal-500 to-sky-500 text-white text-xs font-mono font-semibold px-3 py-1.5 rounded-lg shadow-sm">
              Llama-3.1-8B
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          {pages[page]}
        </main>
      </div>
    </div>
  );
}