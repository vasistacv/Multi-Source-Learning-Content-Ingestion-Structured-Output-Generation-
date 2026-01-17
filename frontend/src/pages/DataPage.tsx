import { useEffect, useState } from 'react';
import { getTasks, uploadFile, processFile } from '../services/api';
import { FileText, Film, Download, CheckCircle, XCircle, Clock, Search, Upload } from 'lucide-react';

const DataPage = () => {
    const [tasks, setTasks] = useState<any[]>([]);
    const [filter, setFilter] = useState('');

    useEffect(() => {
        getTasks().then(res => setTasks(res.tasks));
    }, []);

    const getIcon = (path: string) => {
        if (path?.endsWith('.mp4')) return <Film size={20} className="text-pink-400" />;
        return <FileText size={20} className="text-blue-400" />;
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'text-green-400 bg-green-900/20 border-green-500/30';
            case 'failed': return 'text-red-400 bg-red-900/20 border-red-500/30';
            case 'processing': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30';
            default: return 'text-gray-400 bg-gray-800 border-gray-700';
        }
    };

    const filteredTasks = tasks.filter(t => t.file_path?.toLowerCase().includes(filter.toLowerCase()));

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold title-gradient">Data Sources</h1>
                <div className="flex items-center space-x-4">
                    <div className="relative">
                        <Search className="absolute left-3 top-2.5 text-gray-500" size={18} />
                        <input
                            type="text"
                            placeholder="Filter sources..."
                            className="pl-10 pr-4 py-2 bg-surfaceLight border border-white/10 rounded-lg focus:outline-none focus:border-indigo-500 text-sm w-64"
                            value={filter}
                            onChange={e => setFilter(e.target.value)}
                        />
                    </div>
                    <label className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg cursor-pointer transition-colors">
                        <Upload size={18} />
                        <span className="text-sm font-medium">Ingest New File</span>
                        <input
                            type="file"
                            className="hidden"
                            onChange={async (e) => {
                                if (e.target.files?.[0]) {
                                    try {
                                        const file = e.target.files[0];
                                        // Upload
                                        const uploadRes = await uploadFile(file);
                                        // Process
                                        await processFile(uploadRes.file_id);
                                        // Refresh list
                                        getTasks().then(res => setTasks(res.tasks));
                                        alert('Ingestion started for: ' + file.name);
                                    } catch (err) {
                                        console.error(err);
                                        alert('Upload failed');
                                    }
                                }
                            }}
                        />
                    </label>
                </div>
            </div>

            <div className="glass-panel rounded-xl overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-white/5 border-b border-white/5 text-xs uppercase tracking-wider text-gray-400">
                            <th className="p-4 font-medium">Source Artifact</th>
                            <th className="p-4 font-medium">Ingestion ID</th>
                            <th className="p-4 font-medium">Timestamp</th>
                            <th className="p-4 font-medium">Status</th>
                            <th className="p-4 font-medium text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {filteredTasks.map((task) => (
                            <tr key={task.task_id} className="hover:bg-white/5 transition-colors group">
                                <td className="p-4 flex items-center space-x-3">
                                    <div className="p-2 rounded-lg bg-white/5">
                                        {getIcon(task.file_path)}
                                    </div>
                                    <span className="font-medium text-gray-200 truncate max-w-[200px]">{task.file_path?.split('\\').pop()?.split('/').pop()}</span>
                                </td>
                                <td className="p-4 text-xs font-mono text-gray-500">{task.task_id.slice(0, 8)}...</td>
                                <td className="p-4 text-sm text-gray-400">{new Date(task.created_at).toLocaleDateString()}</td>
                                <td className="p-4">
                                    <span className={`px-2 py-1 rounded-full text-xs border ${getStatusColor(task.status)} flex items-center w-fit space-x-1`}>
                                        {task.status === 'completed' && <CheckCircle size={12} />}
                                        {task.status === 'failed' && <XCircle size={12} />}
                                        {task.status === 'processing' && <Clock size={12} className="animate-spin" />}
                                        <span className="capitalize">{task.status}</span>
                                    </span>
                                </td>
                                <td className="p-4 text-right">
                                    <button className="p-2 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors">
                                        <Download size={18} />
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {filteredTasks.length === 0 && (
                    <div className="p-8 text-center text-gray-500">No data sources found matching "{filter}"</div>
                )}
            </div>
        </div>
    );
};

export default DataPage;
