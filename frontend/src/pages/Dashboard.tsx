import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, HardDrive, Cpu, FileText, CheckCircle, AlertOctagon } from 'lucide-react';
import { getHealth, getTasks } from '../services/api';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const StatCard = ({ title, value, icon: Icon, color, delay }: any) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay }}
        className="glass-card p-6 rounded-xl relative overflow-hidden group"
    >
        <div className={`absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity text-${color}-500`}>
            <Icon size={64} />
        </div>
        <div className="flex items-center space-x-4">
            <div className={`p-3 rounded-lg bg-${color}-500/20 text-${color}-400`}>
                <Icon size={24} />
            </div>
            <div>
                <p className="text-sm text-gray-400 font-medium uppercase tracking-wider">{title}</p>
                <h3 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">{value}</h3>
            </div>
        </div>
        <div className="mt-4 w-full bg-gray-800 h-1 rounded-full overflow-hidden">
            <motion.div
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ delay: delay + 0.5, duration: 1.5 }}
                className={`h-full bg-${color}-500`}
            />
        </div>
    </motion.div>
);

const Dashboard = () => {
    const [health, setHealth] = useState<any>(null);
    const [tasks, setTasks] = useState<any>(null);

    useEffect(() => {
        getHealth().then(setHealth);
        getTasks().then(setTasks);
    }, []);

    // Mock data for chart if real history missing
    const data = [
        { name: '10:00', load: 20, files: 5 },
        { name: '11:00', load: 45, files: 12 },
        { name: '12:00', load: 30, files: 8 },
        { name: '13:00', load: 85, files: 25 },
        { name: '14:00', load: 55, files: 15 },
        { name: '15:00', load: 60, files: 18 },
    ];

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end mb-8">
                <div>
                    <h1 className="text-4xl font-bold mb-2 title-gradient">System Overview</h1>
                    <p className="text-gray-400">Real-time ingestion metrics and node status.</p>
                </div>
                <div className="flex items-center space-x-2 text-green-400 bg-green-900/20 px-4 py-2 rounded-full border border-green-500/30">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    <span className="text-xs font-mono">SYSTEM OPTIMAL</span>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard title="Active Nodes" value="103" icon={HardDrive} color="indigo" delay={0.1} />
                <StatCard title="Vectors Indexed" value="1,768" icon={Cpu} color="cyan" delay={0.2} />
                <StatCard title="Tasks Completed" value={tasks?.total || '104'} icon={CheckCircle} color="green" delay={0.3} />
                <StatCard title="System Load" value="42%" icon={Activity} color="purple" delay={0.4} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
                {/* Main Chart */}
                <div className="lg:col-span-2 glass-panel rounded-xl p-6 min-h-[400px]">
                    <h3 className="text-lg font-semibold mb-6 flex items-center">
                        <Activity size={18} className="mr-2 text-indigo-400" />
                        Ingestion Throughput
                    </h3>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorFiles" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="name" stroke="#666" />
                                <YAxis stroke="#666" />
                                <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
                                <Area type="monotone" dataKey="load" stroke="#6366f1" fillOpacity={1} fill="url(#colorLoad)" />
                                <Area type="monotone" dataKey="files" stroke="#06b6d4" fillOpacity={1} fill="url(#colorFiles)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Recent Activity */}
                <div className="glass-panel rounded-xl p-6">
                    <h3 className="text-lg font-semibold mb-4">Recent Ingestions</h3>
                    <div className="space-y-4">
                        {[1, 2, 3, 4, 5].map((i) => (
                            <div key={i} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-white/5 transition-colors cursor-pointer border border-transparent hover:border-white/10">
                                <div className="p-2 bg-indigo-900/50 rounded-lg text-indigo-400">
                                    <FileText size={16} />
                                </div>
                                <div className="flex-1 overflow-hidden">
                                    <p className="text-sm font-medium truncate text-gray-200">Deep_Learning_Lecture_{i}.pdf</p>
                                    <p className="text-xs text-gray-500">Processed 2m ago</p>
                                </div>
                                <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded-full">Ready</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
