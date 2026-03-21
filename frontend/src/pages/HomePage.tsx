import { Link } from 'react-router-dom';

const features = [
  {
    icon: '🎯',
    title: '模拟面试',
    desc: '语音交互 · AI 实时追问 · 自动评估回答深度',
    to: '/interview',
    cta: '开始面试',
  },
  {
    icon: '📚',
    title: '知识库',
    desc: '导入 EPUB 八股文 · 自动切片入库 · 面试时 RAG 实时召回',
    to: '/knowledge',
    cta: '管理知识库',
  },
];

export default function HomePage() {
  return (
    <div className="home-page">
      <div className="home-hero">
        <h1 className="home-title">Interview Me</h1>
        <p className="home-subtitle">AI 技术面试官 · 语音驱动 · 知识库增强</p>
      </div>
      <div className="home-cards">
        {features.map((f) => (
          <div key={f.to} className="home-card">
            <span className="home-card-icon">{f.icon}</span>
            <h2 className="home-card-title">{f.title}</h2>
            <p className="home-card-desc">{f.desc}</p>
            <Link to={f.to} className="btn btn--primary">{f.cta}</Link>
          </div>
        ))}
      </div>
    </div>
  );
}
