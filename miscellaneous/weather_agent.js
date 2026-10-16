import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// 1. The actual tool implementation
async function getWeather({ location }) {
  // Convert the location name into coordinates
  const geoResponse = await fetch(
    `https://geocoding-api.open-meteo.com/v1/search?` +
      new URLSearchParams({
        name: location,
        count: "1",
      })
  );

  const geoData = await geoResponse.json();
  const place = geoData.results?.[0];

  if (!place) {
    throw new Error(`Could not find location: ${location}`);
  }

  // Get the weather using those coordinates
  const weatherResponse = await fetch(
    `https://api.open-meteo.com/v1/forecast?` +
      new URLSearchParams({
        latitude: String(place.latitude),
        longitude: String(place.longitude),
        current: "temperature_2m,precipitation,wind_speed_10m",
        temperature_unit: "fahrenheit",
      })
  );

  const weatherData = await weatherResponse.json();

  return {
    location: `${place.name}, ${place.admin1 ?? place.country}`,
    temperature: weatherData.current.temperature_2m,
    temperatureUnit: weatherData.current_units.temperature_2m,
    precipitation: weatherData.current.precipitation,
    windSpeed: weatherData.current.wind_speed_10m,
  };
}

// 2. The schema shown to Claude
const tools = [
  {
    name: "get_weather",
    description: "Get the current weather for a location.",
    input_schema: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "A city or location, such as Nashville, Tennessee",
        },
      },
      required: ["location"],
    },
  },
];

// 3. Start the conversation
const messages = [
  {
    role: "user",
    content: "What is the weather in Nashville right now?",
  },
];

let response = await anthropic.messages.create({
  model: "claude-sonnet-4-5",
  max_tokens: 1024,
  tools,
  messages,
});

// 4. Keep handling tool requests until Claude gives a final answer
while (response.stop_reason === "tool_use") {
  messages.push({
    role: "assistant",
    content: response.content,
  });

  const toolResults = [];

  for (const block of response.content) {
    if (block.type === "tool_use" && block.name === "get_weather") {
      try {
        const result = await getWeather(block.input);

        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: JSON.stringify(result),
        });
      } catch (error) {
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          is_error: true,
          content: error.message,
        });
      }
    }
  }

  messages.push({
    role: "user",
    content: toolResults,
  });

  response = await anthropic.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 1024,
    tools,
    messages,
  });
}

// 5. Print Claude's final answer
const finalText = response.content
  .filter((block) => block.type === "text")
  .map((block) => block.text)
  .join("\n");

console.log(finalText);