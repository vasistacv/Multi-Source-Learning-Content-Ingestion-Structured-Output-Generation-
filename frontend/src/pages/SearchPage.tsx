import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles, BookOpen } from 'lucide-react';
import { queryContent } from '../services/api';

const Message = ({ role, content, sources, isTyping }: any) => (
    <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex space-x-4 mb-6 ${role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
        {role === 'assistant' && (
            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                <Bot size={20} className="text-white" />
            </div>
        )}

        <div className={`flex flex-col max-w-[80%] ${role === 'user' ? 'items-end' : 'items-start'}`}>
            <div className={`p-4 rounded-2xl shadow-md ${role === 'user'
                ? 'bg-indigo-600 text-white rounded-br-none'
                : 'glass-card text-gray-200 rounded-bl-none border border-white/5'
                }`}>
                {isTyping ? (
                    <div className="flex space-x-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                    </div>
                ) : (
                    <p className="leading-relaxed whitespace-pre-wrap">{content}</p>
                )}
            </div>

            {sources && sources.length > 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-2 flex flex-wrap gap-2"
                >
                    <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1 w-full">Sources Verified</div>
                    {sources.map((s: any, idx: number) => (
                        <div key={idx} className="flex items-center space-x-1 px-2 py-1 bg-white/5 rounded text-xs text-indigo-300 border border-white/5 hover:bg-white/10 cursor-pointer transition-colors">
                            <BookOpen size={12} />
                            <span className="truncate max-w-[150px]">{s.source}</span>
                            <span className="opacity-50">{(s.relevance_score * 100).toFixed(0)}%</span>
                        </div>
                    ))}
                </motion.div>
            )}
        </div>

        {role === 'user' && (
            <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
                <User size={20} className="text-gray-300" />
            </div>
        )}
    </motion.div>
);

const SearchPage = () => {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState<any[]>([
        { role: 'assistant', content: 'Welcome to your Enterprise Knowledge Hub. How can I assist you with your ingested data today?' }
    ]);
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        const userMsg = { role: 'user', content: query };
        setMessages(prev => [...prev, userMsg]);
        setQuery('');
        setLoading(true);

        try {
            const result = await queryContent(userMsg.content);

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: result.answer,
                sources: result.sources
            }]);
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "I encountered an error connecting to the backend. Please verify the API is running."
            }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col h-[calc(100vh-100px)]">
            <div className="flex-1 overflow-y-auto px-4 py-2" ref={scrollRef}>
                {messages.map((msg, idx) => (
                    <Message key={idx} {...msg} />
                ))}
                {loading && <Message role="assistant" isTyping={true} />}
            </div>

            <div className="p-4 bg-background/50 border-t border-white/5 backdrop-blur-sm sticky bottom-0">
                <form onSubmit={handleSearch} className="relative max-w-4xl mx-auto">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Query the enterprise knowledge base..."
                        className="w-full bg-surfaceLight/50 border border-white/10 rounded-xl px-6 py-4 pr-16 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-white placeholder-gray-500 shadow-xl"
                    />
                    <button
                        type="submit"
                        disabled={loading || !query}
                        className="absolute right-3 top-3 p-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? <Sparkles size={20} className="animate-spin-slow" /> : <Send size={20} />}
                    </button>
                </form>
                <p className="text-center text-[10px] text-gray-500 mt-2 font-mono">
                    POWERED BY FAISS VECTOR INDEX • LLAMA-3-70B • 1768 VECTORS
                </p>
            </div>
        </div>
    );
};

export default SearchPage;
