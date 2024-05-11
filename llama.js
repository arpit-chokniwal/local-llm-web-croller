import "cheerio";
import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { PromptTemplate } from "@langchain/core/prompts";
import { HtmlToTextTransformer } from "@langchain/community/document_transformers/html_to_text";

const PAGE_URL = "https://akshaygugnani.github.io/#page-top"

// Instantiating Ollama
const MODEL = "llama3";
const ollama = new Ollama({ model: MODEL });


// Pipe prompt template to Ollama
const promptTemplate = PromptTemplate.fromTemplate(
    `Answer in detail the questions, based on the context below. If you can't 
    answer the question, reply "I don't know". 

    Context: {detail} 
    Question: {question}`
);
const chain = promptTemplate.pipe(ollama);


// Parsing HTML to text and splitting page into chunks
const loader = new CheerioWebBaseLoader(PAGE_URL);
const docs = await loader.load();
const splitter = RecursiveCharacterTextSplitter.fromLanguage("html", {
    chunkSize: 1500,
    chunkOverlap: 100,
});
const transformer = new HtmlToTextTransformer();
const sequence = splitter.pipe(transformer);
const allSplits = await sequence.invoke(docs);


// Generating embeddings of full documents
const embeddings = new OllamaEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits,
    embeddings
);
const retriever = vectorStore.asRetriever()


const callOllama = async (detail, question) => {
    console.log(`Answer:`);
    const stream = await chain.stream({ detail, question });
    for await (const chunk of stream) {
        process.stdout.write(chunk);
    }
    console.log(`\n\n`);
}

// Getting required details from web page using embeddings according to questions and calling Ollama
const questions = ["Who is this guy?", "Where did he study from?", "Where did he work?", "What are candidate skills?"];
for (const question of questions) {
    console.log(`Question: ${question} \n`);
    const details = await retriever.invoke(question);
    await callOllama(details.map(d => d.pageContent), question);
}
