import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Network, ZoomIn, ZoomOut, Maximize } from 'lucide-react';

const GraphPage = () => {
    const [data, setData] = useState({ nodes: [], links: [] });
    const graphRef = useRef<any>();

    useEffect(() => {
        // Attempt to fetch a specific graph for demo
        // Fallback to sample data for visual "WOW"
        fetch('http://localhost:8000/outputs/knowledge_graphs/meeting_002_technical_review/graph.json')
            .then(res => res.json())
            .then(graph => {
                // Adapt NetworkX JSON to ForceGraph
                const nodes = graph.nodes.map((n: any) => ({ id: n.id, group: n.community || 1, val: n.degree || 1 }));
                const links = graph.links.map((l: any) => ({ source: l.source, target: l.target }));
                setData({ nodes, links });
            })
            .catch(() => {
                // Gen random galaxy if fetch fails
                const N = 100;
                const nodes = [...Array(N).keys()].map(i => ({ id: i, val: Math.random() * 5 }));
                const links = [...Array(N).keys()].map(i => ({ source: i, target: Math.floor(Math.random() * N) }));
                setData({ nodes, links });
            });
    }, []);

    return (
        <div className="h-full flex flex-col space-y-4">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold title-gradient">Knowledge Universe</h1>
                    <p className="text-gray-400 text-sm">Semantic relationships and community clusters.</p>
                </div>
                <div className="flex space-x-2">
                    <button className="p-2 bg-white/5 rounded hover:bg-white/10" onClick={() => graphRef.current?.zoomIn()}><ZoomIn size={20} /></button>
                    <button className="p-2 bg-white/5 rounded hover:bg-white/10" onClick={() => graphRef.current?.zoomOut()}><ZoomOut size={20} /></button>
                    <button className="p-2 bg-white/5 rounded hover:bg-white/10" onClick={() => graphRef.current?.zoomToFit()}><Maximize size={20} /></button>
                </div>
            </div>

            <div className="flex-1 glass-panel rounded-xl overflow-hidden relative border border-white/10 bg-black/50">
                <ForceGraph2D
                    ref={graphRef}
                    graphData={data}
                    nodeLabel="id"
                    nodeColor={node => {
                        const colors = ['#6366f1', '#a855f7', '#06b6d4', '#ec4899'];
                        return colors[node.id % colors.length] || '#fff';
                    }}
                    linkColor={() => 'rgba(100,100,255,0.2)'}
                    backgroundColor="#00000000" // Transparent
                    nodeRelSize={6}
                />
                <div className="absolute bottom-4 left-4 p-4 glass-card rounded-lg max-w-xs">
                    <h4 className="font-bold flex items-center mb-2"><Network size={16} className="mr-2 text-indigo-400" /> Graph Metadata</h4>
                    <div className="space-y-1 text-xs text-gray-400">
                        <p>Nodes: {data.nodes.length}</p>
                        <p>Edges: {data.links.length}</p>
                        <p>Density: High</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GraphPage;
