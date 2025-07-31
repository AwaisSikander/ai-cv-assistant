// server.js (Converted to CommonJS)
const express = require('express');
const cors = require('cors');
require('dotenv/config');

// LangChain imports
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { RunnableSequence, RunnablePassthrough } = require("@langchain/core/runnables");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { Document } = require("@langchain/core/documents");

// Import your CV data
const { cvText } = require('./cv-data.js');

const app = express();

const corsOptions = {
  origin: 'https://askawais.com',
  optionsSuccessStatus: 200
};
app.use(cors(corsOptions));
app.use(express.json());

async function main() {
    // --- Setup LangChain RAG (Retrieval-Augmented Generation) ---
    const model = new ChatGoogleGenerativeAI({
        apiKey: process.env.GOOGLE_GEMINI_API_KEY,
        // --- FIX: Using the latest recommended model name ---
        model: "gemini-2.5-flash",
        maxOutputTokens: 2048,
    });

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GOOGLE_GEMINI_API_KEY,
        model: "embedding-001",
    });

    const document = new Document({ pageContent: cvText });
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 100 });
    const docs = await textSplitter.splitDocuments([document]);
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
    const retriever = vectorStore.asRetriever();

    const template = `You are an expert AI assistant for Awais Sikander. Answer the user's question based only on the following context from his CV. If you don't know the answer from the context, just say you don't have that information. Keep the answers concise.

    CONTEXT: {context}

    QUESTION: {question}

    ANSWER:`;

    const prompt = PromptTemplate.fromTemplate(template);
    
    const chain = RunnableSequence.from([
      {
        context: (input) => retriever.invoke(input.question),
        question: new RunnablePassthrough(),
      },
      {
        context: (previousStepResult) => previousStepResult.context.map((doc) => doc.pageContent).join("\n\n"),
        question: (previousStepResult) => previousStepResult.question.question,
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);
    
    app.post('/ask', async (req, res) => {
        const { question } = req.body;
        if (!question) {
            return res.status(400).json({ error: 'Question is required.' });
        }
        try {
            const result = await chain.invoke({ question });
            res.json({ answer: result });
        } catch (error) {
            console.error('Error processing LangChain request:', error);
            res.status(500).json({ error: 'Failed to process your request.' });
        }
    });

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`AI server is running on http://localhost:${PORT}`);
    });
}

main();