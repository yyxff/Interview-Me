import { NavLink, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import InterviewPage from './pages/InterviewPage';
import KnowledgePage from './pages/KnowledgePage';
import QAPage from './pages/QAPage';
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
            知识库
          </NavLink>
          <NavLink to="/qa" className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}>
            知识问答
          </NavLink>
        </div>
      </nav>

      <div className="page-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/interview" element={<InterviewPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/qa" element={<QAPage />} />
        </Routes>
      </div>
    </div>
  );
}
