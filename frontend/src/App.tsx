import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  Network,
  Database,
  BookOpen,
  Settings,
  Menu,
  X,
  Cpu
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Pages (Placeholder imports for now, we will create them next)
import Dashboard from './pages/Dashboard';
import SearchPage from './pages/SearchPage';
import GraphPage from './pages/GraphPage';
import DataPage from './pages/DataPage';

const SidebarItem = ({ to, icon: Icon, label, active }: any) => (
  <Link to={to}>
    <div className={`flex items-center space-x-3 px-4 py-3 rounded-lg mb-1 transition-all duration-200 ${active
      ? 'bg-primary/20 text-indigo-300 border-r-2 border-indigo-500'
      : 'text-gray-400 hover:bg-white/5 hover:text-white'
      }`}>
      <Icon size={20} />
      <span className="font-medium">{label}</span>
      {active && <motion.div layoutId="active-pill" className="absolute right-0 w-1 h-8 bg-indigo-500 rounded-l-full" />}
    </div>
  </Link>
);

const Layout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Mission Control' },
    { path: '/search', icon: Search, label: 'Neural Search' },
    { path: '/graph', icon: Network, label: 'Knowledge Universe' },
    { path: '/data', icon: Database, label: 'Data Sources' },
    // { path: '/artifacts', icon: BookOpen, label: 'Learning Artifacts' },
  ];

  return (
    <div className="flex h-screen bg-background text-white overflow-hidden">
      {/* Sidebar */}
      <motion.aside
        initial={{ x: -250 }}
        animate={{ x: 0 }}
        className="hidden md:flex flex-col w-64 glass-panel border-r border-white/10 z-20"
      >
        <div className="p-6 flex items-center space-x-3 border-b border-white/5">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/20">
            <Cpu size={24} className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-wide">KNOWLEDGE HUB</h1>
            <p className="text-xs text-indigo-400 font-mono">Enterprise Edition</p>
          </div>
        </div>

        <nav className="flex-1 px-3 py-6 overflow-y-auto">
          <p className="px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Modules</p>
          {navItems.map((item) => (
            <SidebarItem
              key={item.path}
              to={item.path}
              icon={item.icon}
              label={item.label}
              active={location.pathname === item.path}
            />
          ))}
        </nav>

        <div className="p-4 border-t border-white/5">
          <div className="glass-card p-4 rounded-xl">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-gray-400">System Status</span>
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5 mb-2">
              <div className="bg-indigo-500 h-1.5 rounded-full w-[35%]"></div>
            </div>
            <p className="text-[10px] text-gray-500 font-mono">GPU: ACTIVE • MEM: OPTIMIZED</p>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-gradient-to-br from-background via-[#0c0c1e] to-background">
        {/* Header */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-6 glass-panel z-10 backdrop-blur-md">
          <div className="flex items-center">
            <button className="md:hidden mr-4 text-gray-400" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
              <Menu size={24} />
            </button>
            <h2 className="text-xl font-semibold text-gray-200">
              {navItems.find(i => i.path === location.pathname)?.label || 'Dashboard'}
            </h2>
          </div>
          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
              <span className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></span>
              <span className="text-xs font-mono text-gray-300">GROQ-70B CONNECTED</span>
            </div>
            <div className="p-2 bg-white/5 rounded-full hover:bg-white/10 cursor-pointer">
              <Settings size={20} className="text-gray-400" />
            </div>
          </div>
        </header>

        {/* Content View */}
        <div className="flex-1 overflow-auto p-6 md:p-8 relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/data" element={<DataPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
