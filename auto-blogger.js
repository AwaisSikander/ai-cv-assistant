// auto-blogger.js
const axios = require("axios");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// --- LangChain Imports ---
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
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

// --- LangChain Setup for Content Generation ---
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_GEMINI_API_KEY,
  model: "gemini-2.5-flash",
});

const promptTemplate = PromptTemplate.fromTemplate(
  `You are Awais Sikander, a Lead Full Stack Developer and a Blockchain expert. Your blog, askawais.com, provides in-depth technical tutorials and insights for other developers.
    
    Generate a complete, high-quality, and SEO-friendly blog post about "{topic}". The post must be written in a professional yet approachable tone, as if it were written by a seasoned developer.
    
    Ensure the content is well-structured with clear headings (e.g., <h2>, <h3>) and valid HTML markup. The "body" should be a single string containing all HTML. Do not use Markdown outside of the JSON object.
    
    Provide the output as a single, clean JSON object.
    
    Example JSON structure:
    {{
      "title": "Your Blog Post Title Here",
      "body": "<p>This is the first paragraph...</p><h2>A Sub-heading</h2><p>Content for the sub-heading.</p>",
      "excerpt": "A short, 1-2 sentence summary of the post."
    }}
    
    Strictly follow this JSON format. All values must be strings.
    
    Post Topic: {topic}
    
    JSON Output:`
);

const generationChain = RunnableSequence.from([
  {
    topic: new RunnablePassthrough(),
  },
  promptTemplate,
  model,
  new StringOutputParser(),
]);
// --- End of LangChain Setup ---

// ⬇️ THE FINAL FIX IS IN THIS FUNCTION ⬇️
async function generateContent(topic, log) {
  log(`Generating content for topic: ${topic}`);
  try {
    const output = await generationChain.invoke({ topic: topic });

    // ✅ FIX: Use a regular expression to clean up the JSON string before parsing.
    const cleanOutput = output
      .replace(/```json/g, "")
      .replace(/```/g, "")
      .trim();

    const jsonStart = cleanOutput.indexOf("{");
    const jsonEnd = cleanOutput.lastIndexOf("}") + 1;
    const jsonString = cleanOutput.substring(jsonStart, jsonEnd);

    // ✅ FIX: Replace bad control characters inside the string
    const sanitizedJsonString = jsonString.replace(
      /[\u0000-\u001F\u007F-\u009F]/g,
      ""
    );

    const result = JSON.parse(sanitizedJsonString);
    return result;
  } catch (error) {
    log(`Error generating content: ${error.message}`);
    console.error("Full content generation error:", error);
    return null;
  }
}
// ⬆️ THE FINAL FIX IS IN THIS FUNCTION ⬆️

async function generateAndUploadImage(title, log) {
  log(`Adding title to default background: "${title}"`);
  try {
    const backgroundImagePath = path.join(__dirname, "./background.png");
    if (!fs.existsSync(backgroundImagePath)) {
      throw new Error(`Background image not found at ${backgroundImagePath}`);
    }
    const width = 1920;
    const height = 1080;

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
      return lines
        .map(
          (line, index) =>
            `<tspan x="50%" dy="${index === 0 ? 0 : "1.2em"}">${line}</tspan>`
        )
        .join("");
    }

    const wrappedTitle = wrapText(escapedTitle, 30);
    const svgText = `
      <svg width="${width}" height="${height}">
        <style>
          .title { font-family: 'Helvetica', 'Verdana', sans-serif; font-size: 90px; font-weight: bold; fill: #333333; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        </style>
        <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" class="title">
          ${wrappedTitle}
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

async function createPost(postContent, imageId, log) {
  log(`Creating post: ${postContent.title}`);
  try {
    const response = await axios.post(
      `${WP_URL}/wp-json/wp/v2/posts`,
      {
        title: postContent.title,
        content: postContent.body,
        excerpt: postContent.excerpt,
        status: "publish",
        featured_media: imageId,
      },
      { headers: wpHeaders }
    );
    log(`✅ Post published successfully! URL: ${response.data.link}`);
  } catch (error) {
    log(
      `Error creating post: ${
        error.response ? JSON.stringify(error.response.data) : error.message
      }`
    );
  }
}

async function main(log = console.log) {
  log(`🚀 Starting the auto-blogger job at ${new Date().toLocaleString()}`);
  const topics = [
    "Building a simple DAO with Solidity",
    "Microfrontend architecture deep dive",
    "Getting started with AWS Lambda for Node.js developers",
  ];
  const randomTopic = topics[Math.floor(Math.random() * topics.length)];

  const postContent = await generateContent(randomTopic, log);
  if (postContent && postContent.title) {
    const imageId = await generateAndUploadImage(postContent.title, log);
    if (imageId) {
      await createPost(postContent, imageId, log);
    } else {
      log("Skipping post creation due to image generation failure.");
    }
  } else {
    log("Skipping post creation due to content generation failure.");
  }
  log("✨ Blogger job finished.");
}

module.exports = { main };
