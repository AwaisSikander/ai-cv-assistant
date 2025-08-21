// auto-blogger.js
require("dotenv").config();
const axios = require("axios");
const { GoogleGenAI } = require("@google/genai");

// --- Configuration ---
const WP_URL = process.env.WP_URL;
const WP_USER = process.env.WP_USER;
const WP_PASSWORD = process.env.WP_PASSWORD;

// --- Initialize Google AI Client ---
const genAI = new GoogleGenAI(process.env.GOOGLE_GEMINI_API_KEY);

// WordPress authentication
const credentials = Buffer.from(`${WP_USER}:${WP_PASSWORD}`).toString("base64");
const wpHeaders = { Authorization: `Basic ${credentials}` };

async function generateContent(topic) {
  // ... (This function remains unchanged)
  console.log(`Generating content for topic: ${topic}`);
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
  const prompt = `You are a tech blogger for askawais.com. Generate a complete, SEO-friendly blog post about "${topic}". Provide the output in a single, clean JSON object with keys: "title", "body", "excerpt", "image_prompt".`;
  try {
    const result = await model.generateContent(prompt);
    const rawText = result.response.text();
    const textWithoutJsonTag = rawText.replace(/```json/g, "");
    const textWithoutBackticks = textWithoutJsonTag.replace(/```/g, "");
    const jsonString = textWithoutBackticks.trim();
    return JSON.parse(jsonString);
  } catch (error) {
    console.error("Error generating content:", error);
    return null;
  }
}

async function generateAndUploadImage(prompt) {
  // ... (This function remains unchanged)
  console.log(`Generating image with Imagen for prompt: "${prompt}"`);
  try {
    const response = await genAI.models.generateImages({
      model: "imagen-4.0-fast-generate-001",
      prompt: prompt,
      config: { numberOfImages: 1, aspectRatio: "16:9" },
    });
    const generatedImage = response.generatedImages[0];
    const imageBuffer = Buffer.from(generatedImage.image.imageBytes, "base64");
    const imageName = `imagen-ai-image-${Date.now()}.png`;
    console.log("Uploading generated image to WordPress...");
    const mediaResponse = await axios.post(
      `${WP_URL}/wp-json/wp/v2/media`,
      imageBuffer,
      {
        headers: {
          ...wpHeaders,
          "Content-Type": "image/png",
          "Content-Disposition": `attachment; filename="${imageName}"`,
        },
      }
    );
    console.log(
      `Image uploaded successfully. Media ID: ${mediaResponse.data.id}`
    );
    return mediaResponse.data.id;
  } catch (error) {
    console.error("Error generating or uploading image:", error);
    return null;
  }
}

async function createPost(postContent, imageId) {
  // ... (This function remains unchanged)
  console.log(`Creating post: ${postContent.title}`);
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
    console.log(`✅ Post published successfully! URL: ${response.data.link}`);
  } catch (error) {
    console.error(
      "Error creating post:",
      error.response ? error.response.data : error.message
    );
  }
}

// --- Main Execution Logic ---
async function main() {
  console.log(
    `🚀 Starting the auto-blogger job at ${new Date().toLocaleString()}`
  );
  const topics = [
    "Building a simple DAO with Solidity",
    "Microfrontend architecture deep dive",
    "Advanced JavaScript techniques for 2025",
    "Performance optimization in Node.js applications",
  ];
  const randomTopic = topics[Math.floor(Math.random() * topics.length)];
  const postContent = await generateContent(randomTopic);
  if (postContent) {
    const imageId = await generateAndUploadImage(postContent.image_prompt);
    await createPost(postContent, imageId);
  }
  console.log("✨ Blogger job finished successfully.");
}

// ⬇️ NEW: Export the main function so other files can use it
module.exports = { main };
