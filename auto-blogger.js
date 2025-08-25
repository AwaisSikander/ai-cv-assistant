// auto-blogger.js
const axios = require("axios");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// --- LangChain Imports ---
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
const { JsonOutputParser } = require("@langchain/core/output_parsers");
const {
  RunnableSequence,
  RunnablePassthrough,
} = require("@langchain/core/runnables");

// --- Configuration & Initialization ---
const WP_URL = process.env.WP_URL;
const WP_USER = process.env.WP_USER;
const WP_PASSWORD = process.env.WP_PASSWORD;
const credentials = Buffer.from(`${WP_USER}:${WP_PASSWORD}`).toString("base64");
const wpHeaders = { Authorization: `Basic ${credentials}` };

// --- LangChain Setup ---
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_GEMINI_API_KEY,
  model: "gemini-1.5-flash",
});

// --- AI #1: The Visionary Title Brainstormer ---
const titlePromptTemplate = PromptTemplate.fromTemplate(
  `You are a Principal Engineer and Content Strategist, tasked with generating visionary blog post ideas.

   Your goal is to generate 5 expert-level titles based on the broad technical domain of: "{domain}".

   **Your thought process MUST follow these two steps:**
   1.  **Internal Brainstorming:** First, silently and internally, brainstorm a list of advanced, non-obvious sub-topics within the '{domain}'. Think at a micro and nano level about performance, security, architectural patterns, and future trends. For example, for 'Node.js', you might internally brainstorm 'libuv internals', 'memory leak patterns in async contexts', 'multi-tenancy security models', 'WASI integration'.
   2.  **Title Generation:** Second, using your brainstormed list of advanced sub-topics, generate 5 highly specific, visionary titles that would capture the attention of a Staff-level engineer.

   **CRITICAL INSTRUCTIONS:**
   - AVOID generating titles about the broad domain itself (e.g., "What is Node.js?").
   - AVOID any beginner or intermediate-level concepts.
   - Your final output MUST be only a clean JSON object with a single key "titles" which contains an array of 5 title strings.
   - You MUST avoid generating titles that are thematically similar to the following already published titles:
   <published_titles>
   {published_titles}
   </published_titles>
   `
);

const titleGenerationChain = RunnableSequence.from([
  titlePromptTemplate,
  model,
  new JsonOutputParser(),
]);

// --- AI #2: The Expert Article Writer ---
const articlePromptTemplate = PromptTemplate.fromTemplate(
  `You are Awais Sikander, a Lead Full Stack Developer and Blockchain expert. Your blog, askawais.com, provides elite-level technical articles.
   
   Your task is to write the **full body content** for a blog post, with complete paragraphs.
   The title of the post is: "{title}".

   **CRITICAL INSTRUCTIONS:**
   1.  **Do NOT repeat the main title** in your response. Begin directly with the first paragraph.
   2.  **Do NOT write an outline or just a list of topics.** You must write a complete article with detailed, flowing paragraphs under each subheading.
   3.  The content must be in-depth, professional, and well-structured with HTML subheadings (<h2>, <h3>).
   4.  **Include expert-level code examples** where appropriate to illustrate your points. Wrap all code blocks in \`<pre><code class="language-javascript">...\</code></pre>\` tags, replacing 'javascript' with the appropriate language (e.g., 'solidity', 'php', 'bash').

   - The "excerpt" must be a short and compelling hook. It should pose a thought-provoking question or make a bold statement that makes an expert want to click, read the full article, and comment.  

   Provide the output as a single, clean JSON object with "body" and "excerpt" keys.
   
   JSON Output:`
);

const articleGenerationChain = RunnableSequence.from([
  articlePromptTemplate,
  model,
  new StringOutputParser(),
]);

async function generateContent(title, log) {
  log(`Generating content for title: ${title}`);
  try {
    const output = await articleGenerationChain.invoke({ title: title });
    const jsonStart = output.indexOf("{");
    const jsonEnd = output.lastIndexOf("}") + 1;
    const jsonString = output.substring(jsonStart, jsonEnd);
    const result = JSON.parse(jsonString);
    return result;
  } catch (error) {
    log(`Error generating content: ${error.message}`);
    return null;
  }
}

async function generateAndUploadImage(title, log) {
  log(`Adding title to default background: "${title}"`);
  try {
    const backgroundImagePath = path.join(__dirname, "./background.png");
    if (!fs.existsSync(backgroundImagePath)) {
      throw new Error(`Background image not found at ${backgroundImagePath}`);
    }
    const width = 1920;
    const height = 1080;
    const fontSize = 90;
    const lineHeight = 1.2;

    const escapeXml = (unsafe) => {
      return unsafe.replace(/[<>&'"]/g, (c) => {
        switch (c) {
          case "<":
            return "&lt;";
          case ">":
            return "&gt;";
          case "&":
            return "&amp;";
          case "'":
            return "&apos;";
          case '"':
            return "&quot;";
          default:
            return c;
        }
      });
    };

    const escapedTitle = escapeXml(title);

    function wrapText(text, charsPerLine) {
      const words = text.split(" ");
      let lines = [];
      let currentLine = words[0] || "";
      for (let i = 1; i < words.length; i++) {
        if (currentLine.length + words[i].length + 1 < charsPerLine) {
          currentLine += ` ${words[i]}`;
        } else {
          lines.push(currentLine);
          currentLine = words[i];
        }
      }
      lines.push(currentLine);
      return lines;
    }

    const lines = wrapText(escapedTitle, 30);
    const lineCount = lines.length;

    const tspanElements = lines
      .map(
        (line, index) =>
          `<tspan x="50%" dy="${
            index === 0 ? 0 : `${lineHeight}em`
          }">${line}</tspan>`
      )
      .join("");

    const totalTextBlockHeight = (lineCount - 1) * lineHeight;
    const yOffset = -(totalTextBlockHeight / 2);

    const svgText = `
      <svg width="${width}" height="${height}">
        <style>
          .title { font-family: 'Helvetica', 'Verdana', sans-serif; font-size: ${fontSize}px; font-weight: bold; fill: #333333; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        </style>
        <text x="50%" y="50%" dy="${yOffset}em" text-anchor="middle" class="title">
          ${tspanElements}
        </text>
      </svg>`;

    const imageBuffer = await sharp(backgroundImagePath)
      .composite([{ input: Buffer.from(svgText), gravity: "center" }])
      .jpeg({ quality: 90 })
      .toBuffer();

    const imageName = `featured-image-${Date.now()}.jpg`;
    log("Uploading generated image to WordPress...");
    const mediaResponse = await axios.post(
      `${WP_URL}/wp-json/wp/v2/media`,
      imageBuffer,
      {
        headers: {
          ...wpHeaders,
          "Content-Type": "image/jpeg",
          "Content-Disposition": `attachment; filename="${imageName}"`,
        },
      }
    );
    log(`Image uploaded successfully. Media ID: ${mediaResponse.data.id}`);
    return mediaResponse.data.id;
  } catch (error) {
    log(
      `Error generating or uploading image: ${
        error.response ? JSON.stringify(error.response.data) : error.message
      }`
    );
    return null;
  }
}

// =========================================================================================
// START: ROBUST GUTENBERG BLOCK CONVERTER
// =========================================================================================
function convertToGutenbergBlocks(html) {
  // This regex splits the HTML by the start of a heading or paragraph tag,
  // which is more robust than matching full tags.
  const chunks = html.split(/(?=<h[2-3]|<p)/);

  return chunks
    .map((chunk) => {
      const trimmedChunk = chunk.trim();
      if (!trimmedChunk) {
        return ""; // Ignore empty chunks
      }

      if (trimmedChunk.startsWith("<h")) {
        return `${trimmedChunk}`;
      }

      if (trimmedChunk.startsWith("<p")) {
        return `${trimmedChunk}`;
      }

      // This is the crucial fallback: if a chunk of text is not in a tag,
      // we wrap it in <p> tags and then convert it to a paragraph block.
      return `<p>${trimmedChunk}</p>`;
    })
    .filter(Boolean) // Filter out any empty strings that might have been created
    .join("\n\n");
}
// =========================================================================================
// END: ROBUST GUTENBERG BLOCK CONVERTER
// =========================================================================================

async function createPost(postContent, imageId, log) {
  log(`Creating post: ${postContent.title}`);
  try {
    const response = await axios.post(
      `${WP_URL}/wp-json/wp/v2/posts`,
      {
        title: postContent.title,
        content: postContent.body, // This will now be the robustly-formatted Gutenberg content
        excerpt: postContent.excerpt,
        status: "publish",
        featured_media: imageId,
        categories: postContent.categoryIds,
      },
      { headers: wpHeaders }
    );
    log(`✅ Post published successfully! URL: ${response.data.link}`);
    return true;
  } catch (error) {
    log(
      `Error creating post: ${
        error.response ? JSON.stringify(error.response.data) : error.message
      }`
    );
    return false;
  }
}

// =========================================================================================
// START: MAIN ORCHESTRATION LOGIC
// =========================================================================================
async function main(log = console.log) {
  log(`🚀 Starting the auto-blogger job at ${new Date().toLocaleString()}`);

  const topicDomains = [
    { domain: "Advanced JavaScript", categoryIds: [10] },
    { domain: "React.js ", categoryIds: [8, 18] },
    { domain: "Vue.js ", categoryIds: [4] },
    { domain: "Node.js", categoryIds: [7] },
    { domain: "Laravel", categoryIds: [9, 14] },
    { domain: "Modern PHP", categoryIds: [14] },
    { domain: "Database ", categoryIds: [15] },
    { domain: "Solidity & Smart Contract", categoryIds: [11] },
    { domain: "Advanced Hardhat & Web3 Tooling", categoryIds: [12, 11] },
    { domain: "DeFi & Oracle Technology (Chainlink)", categoryIds: [11] },
    { domain: "Advanced NFT Standards", categoryIds: [11] },
    { domain: "DAO Governance Models", categoryIds: [11, 18] },
    { domain: "Cloud & Serverless Architecture", categoryIds: [18] },
    { domain: "Containerization & CI/CD (Docker)", categoryIds: [18] },
    { domain: "Sass Or Erp", categoryIds: [1] },
    { domain: "Latest Mysql oR Mongodb ", categoryIds: [1] },
    {
      domain: "Vue & React Optimization or Laravel & Nodejs Optimization",
      categoryIds: [1],
    },
    {
      domain: "Tailwindcss or Modern Ui libraries or framework",
      categoryIds: [1],
    },
    {
      domain: "Artificial Intelligence in Web Development",
      categoryIds: [6, 18],
    },
  ];

  const LOG_FILE = path.join(__dirname, "published_titles.log");
  let publishedTitles = [];
  if (fs.existsSync(LOG_FILE)) {
    publishedTitles = fs
      .readFileSync(LOG_FILE, "utf-8")
      .split("\n")
      .filter(Boolean);
  }

  const randomDomainItem =
    topicDomains[Math.floor(Math.random() * topicDomains.length)];
  log(`Selected domain: "${randomDomainItem.domain}"`);

  log("Brainstorming visionary titles from the selected domain...");
  const titleResponse = await titleGenerationChain.invoke({
    domain: randomDomainItem.domain,
    published_titles: publishedTitles.join("\n") || "N/A",
  });

  const uniqueTitles = titleResponse.titles;
  if (!uniqueTitles || uniqueTitles.length === 0) {
    log(
      "Could not brainstorm any new unique titles for this domain. Try again later or add more domains."
    );
    return;
  }
  const selectedTitle =
    uniqueTitles[Math.floor(Math.random() * uniqueTitles.length)];

  const postContent = await generateContent(selectedTitle, log);
  if (postContent) {
    // Convert the raw HTML body to Gutenberg-compatible blocks
    postContent.body = convertToGutenbergBlocks(postContent.body);

    postContent.title = selectedTitle;
    postContent.categoryIds = randomDomainItem.categoryIds;

    const imageId = await generateAndUploadImage(postContent.title, log);
    if (imageId) {
      const postCreated = await createPost(postContent, imageId, log);
      if (postCreated) {
        fs.appendFileSync(LOG_FILE, selectedTitle + "\n");
        log(`✅ Title "${selectedTitle}" has been logged as published.`);
      }
    } else {
      log("Skipping post creation due to image generation failure.");
    }
  } else {
    log("Skipping post creation due to content generation failure.");
  }
  log("✨ Blogger job finished.");
}
// =========================================================================================
// END: MAIN LOGIC
// =========================================================================================

module.exports = { main };

// This check ensures main() only runs when the script is executed directly
if (require.main === module) {
  main();
}
