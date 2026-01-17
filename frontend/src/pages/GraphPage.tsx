import { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Network, ZoomIn, ZoomOut, Maximize, Sparkles } from 'lucide-react';

interface GraphData {
    nodes: { id: string; name: string; val: number; color: string }[];
    links: { source: string; target: string; value: number }[];
}

const COLORS = ['#6366f1', '#a855f7', '#06b6d4', '#ec4899', '#14b8a6', '#f59e0b'];

const GraphPage = () => {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
    const graphRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const getNodeColor = useCallback((index: number) => {
        return COLORS[index % COLORS.length];
    }, []);

    // Load demo data immediately
    useEffect(() => {
        const N = 100;
        const nodes = [...Array(N).keys()].map(i => ({
            id: `concept-${i}`,
            name: `Concept ${i}`,
            val: Math.random() * 6 + 2,
            color: getNodeColor(i)
        }));
        const links = [...Array(N * 2).keys()].map(() => ({
            source: `concept-${Math.floor(Math.random() * N)}`,
            target: `concept-${Math.floor(Math.random() * N)}`,
            value: Math.random()
        }));

        setData({ nodes, links });
        setLoading(false);
    }, [getNodeColor]);

    // Handle resize
    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth || 800,
                    height: containerRef.current.clientHeight || 500
                });
            }
        };

        updateSize();
        window.addEventListener('resize', updateSize);
        const timer = setTimeout(updateSize, 200);

        return () => {
            window.removeEventListener('resize', updateSize);
            clearTimeout(timer);
        };
    }, []);

    return (
        <div className="h-full flex flex-col space-y-4">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold title-gradient flex items-center gap-2">
                        <Sparkles className="animate-pulse" />
                        Knowledge Universe
                    </h1>
                    <p className="text-gray-400 text-sm">Interactive semantic relationship network visualization.</p>
                </div>
                <div className="flex space-x-2">
                    <button className="p-2 bg-white/5 rounded hover:bg-white/10 transition-colors" onClick={() => graphRef.current?.zoomToFit(400)}>
                        <Maximize size={20} />
                    </button>
                    <button className="p-2 bg-white/5 rounded hover:bg-white/10 transition-colors" onClick={() => graphRef.current?.zoom(1.5)}>
                        <ZoomIn size={20} />
                    </button>
                    <button className="p-2 bg-white/5 rounded hover:bg-white/10 transition-colors" onClick={() => graphRef.current?.zoom(0.5)}>
                        <ZoomOut size={20} />
                    </button>
                </div>
            </div>

            <div
                ref={containerRef}
                className="flex-1 glass-panel rounded-xl overflow-hidden relative border border-white/10"
                style={{ minHeight: '500px', backgroundColor: 'rgba(0,0,0,0.8)' }}
            >
                {loading ? (
                    <div className="w-full h-full flex items-center justify-center">
                        <div className="text-center">
                            <Sparkles className="animate-spin w-12 h-12 text-indigo-500 mx-auto mb-4" />
                            <p className="text-gray-400">Rendering Knowledge Graph...</p>
                        </div>
                    </div>
                ) : (
                    <ForceGraph2D
                        ref={graphRef}
                        width={dimensions.width}
                        height={dimensions.height}
                        graphData={data}
                        nodeLabel={(node: any) => node.name}
                        nodeColor={(node: any) => node.color}
                        nodeRelSize={6}
                        nodeVal={(node: any) => node.val}
                        linkColor={() => 'rgba(99, 102, 241, 0.4)'}
                        linkWidth={1}
                        linkDirectionalParticles={2}
                        linkDirectionalParticleSpeed={0.005}
                        linkDirectionalParticleWidth={2}
                        backgroundColor="transparent"
                        warmupTicks={50}
                        cooldownTicks={100}
                    />
                )}

                <div className="absolute bottom-4 left-4 p-4 glass-card rounded-lg">
                    <h4 className="font-bold flex items-center mb-2">
                        <Network size={16} className="mr-2 text-indigo-400" />
                        Graph Metadata
                    </h4>
                    <div className="space-y-1 text-xs text-gray-400">
                        <p>Nodes: <span className="text-white font-mono">{data.nodes.length}</span></p>
                        <p>Edges: <span className="text-white font-mono">{data.links.length}</span></p>
                        <p>Status: <span className="text-green-400">Active</span></p>
                    </div>
                </div>

                <div className="absolute top-4 right-4 p-3 glass-card rounded-lg text-xs text-gray-400">
                    <p className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"></span>
                        Live Simulation
                    </p>
                </div>
            </div>
        </div>
    );
};

export default GraphPage;
