import { NavLink, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import InterviewPage from './pages/InterviewPage';
import InterviewReviewPage from './pages/InterviewReviewPage';
import KnowledgeQAPage from './pages/KnowledgeQAPage';
import './App.css';

export default function App() {
  return (
    <div className="app-shell">
      <nav className="main-nav">
        <NavLink to="/" className="nav-brand" end>
          <span className="logo">🎯</span>
          <span className="nav-brand-title">Interview Me</span>
        </NavLink>
        <div className="nav-links">
          <NavLink to="/" className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`} end>
            首页
          </NavLink>
          <NavLink to="/interview" className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}>
            模拟面试
          </NavLink>
          <NavLink to="/knowledge" className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}>
            知识库 & 问答
          </NavLink>
          <NavLink to="/review" className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}>
            面试复盘
          </NavLink>
        </div>
      </nav>

      <div className="page-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/interview" element={<InterviewPage />} />
          <Route path="/review" element={<InterviewReviewPage />} />
          <Route path="/knowledge" element={<KnowledgeQAPage />} />
          <Route path="/qa" element={<KnowledgeQAPage />} />
        </Routes>
      </div>
    </div>
  );
}
