// server.js
const express = require("express");
const cors = require("cors");
require("dotenv/config");

const { main: runAutoBlogger } = require("./auto-blogger.js");

const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const {
  RunnableSequence,
  RunnablePassthrough,
} = require("@langchain/core/runnables");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { Document } = require("@langchain/core/documents");

const { cvText } = require("./cv-data.js");

const app = express();
const corsOptions = {
  origin: "https://askawais.com",
  optionsSuccessStatus: 200,
};
app.use(cors(corsOptions));
app.use(express.json());

async function initializeApp() {
  const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_GEMINI_API_KEY,
    // --- MODEL UPDATED HERE ---
    model: "gemini-2.5-flash",
  });
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_GEMINI_API_KEY,
    model: "embedding-001",
  });
  const document = new Document({ pageContent: cvText });
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });
  const docs = await textSplitter.splitDocuments([document]);
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const retriever = vectorStore.asRetriever();
  const template = `You are an expert AI assistant for Awais Sikander. Answer the user's question based only on the following context from his CV. If you don't know the answer, say you don't have that information. Keep answers concise. CONTEXT: {context} QUESTION: {question} ANSWER:`;
  const prompt = PromptTemplate.fromTemplate(template);
  const chain = RunnableSequence.from([
    {
      context: (input) => retriever.invoke(input.question),
      question: new RunnablePassthrough(),
    },
    {
      context: (prev) =>
        prev.context.map((doc) => doc.pageContent).join("\n\n"),
      question: (prev) => prev.question.question,
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);

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

  app.get("/run-blogger", async (req, res) => {
    const { secret } = req.query;
    if (secret !== process.env.CRON_SECRET_KEY) {
      return res.status(401).send("Unauthorized");
    }
    res
      .status(202)
      .send(
        "Accepted: Auto-blogger process has been started in the background."
      );
    try {
      await runAutoBlogger();
    } catch (error) {
      console.error("Error running the auto-blogger process:", error);
    }
  });

  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`AI server is running on port ${PORT}`);
  });
}

initializeApp();
