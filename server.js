// server.js
import express from "express";
import cors from "cors";
import "dotenv/config";

// LangChain imports
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
// --- CHANGE: Using the simpler MemoryVectorStore ---
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RunnableSequence } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";

// Import your CV data
import { cvText } from "./cv-data.js";

// --- Initialize Express App ---
const app = express();
app.use(cors());
app.use(express.json());

// --- Setup LangChain RAG (Retrieval-Augmented Generation) ---
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_GEMINI_API_KEY,
  model: "gemini-pro",
  maxOutputTokens: 2048,
});

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_GEMINI_API_KEY,
  model: "embedding-001",
});

// Create a document from your CV text
const document = new Document({ pageContent: cvText });

// Split the document into smaller chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100,
});
const docs = await textSplitter.splitDocuments([document]);

// --- CHANGE: Create a vector store from the chunks using MemoryVectorStore ---
const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
const retriever = vectorStore.asRetriever();

// Create a prompt template
const template = `You are an expert AI assistant for Awais Sikander. Answer the user's question based only on the following context from his CV. If you don't know the answer from the context, just say you don't have that information. Keep the answers concise.

CONTEXT: {context}

QUESTION: {question}

ANSWER:`;

const prompt = PromptTemplate.fromTemplate(template);

// Create the final chain
const chain = RunnableSequence.from([
  {
    context: retriever.pipe((docs) =>
      docs.map((d) => d.pageContent).join("\n\n")
    ),
    question: (input) => input.question,
  },
  prompt,
  model,
  new StringOutputParser(),
]);

// --- API Endpoint ---
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: "Question is required." });
  }

  try {
    const result = await chain.invoke({ question });
    res.json({ answer: result });
  } catch (error) {
    console.error("Error processing LangChain request:", error);
    res.status(500).json({ error: "Failed to process your request." });
  }
});

// --- Start the Server ---
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`AI server is running on http://localhost:${PORT}`);
});
