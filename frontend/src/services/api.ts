import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface Task {
    task_id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    file_id: string;
    created_at: string;
    result?: any;
    error?: string;
    output_path?: string;
}

export interface QueryResponse {
    query: string;
    answer: string;
    sources: {
        source: string;
        relevance_score: number;
        text: string;
    }[];
}

export const getHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

export const getTasks = async (status?: string) => {
    const response = await api.get('/tasks', { params: { status, limit: 50 } });
    return response.data;
};

export const queryContent = async (query: string, top_k: number = 5) => {
    try {
        const response = await api.post<QueryResponse>('/query', { query, top_k });
        return response.data;
    } catch (error) {
        console.error("API Error in queryContent:", error);
        return {
            query: query,
            answer: "", // Empty answer triggers the fallback logic in SearchPage
            sources: []
        };
    }
};

export const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

export const processFile = async (fileId: string) => {
    const response = await api.post('/process', null, {
        params: { file_id: fileId },
    });
    return response.data;
};
