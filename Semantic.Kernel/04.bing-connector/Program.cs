
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Plugins.Web;
using Microsoft.SemanticKernel.Plugins.Web.Bing;
using Kernel = Microsoft.SemanticKernel.Kernel;

var builder = Kernel.CreateBuilder();

// Configure AI backend used by the kernel
var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

if (useAzureOpenAI)
    builder.AddAzureOpenAIChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOpenAIChatCompletion(model, apiKey, orgId);

var kernel = builder.Build();
var question = "How much was the 2024 Q4 revenue for Microsoft?";

Console.WriteLine(question);
Console.WriteLine("-----");

// Section 1: Ask LLM
var prompt = question;
var answer = await kernel.InvokePromptAsync(prompt);
Console.WriteLine(answer.GetValue<string>());

/*
// Section 2: Get the data from Bing Search
string BING_KEY = Settings.BingSearchKey();

// Load Bing plugin
var bingConnector = new BingConnector(BING_KEY);

kernel.ImportPluginFromObject(new WebSearchEnginePlugin(bingConnector), "bing");

var function = kernel.Plugins["bing"]["search"];
var bingResult = await kernel.InvokeAsync(function, new() { ["query"] = question });

Console.WriteLine(bingResult);

// Section 2: Supply the search result to the prompt context
Console.WriteLine();
var prompt = $@"
        Statement: {bingResult}
        From above Statement: {question}
        ";
var answer = await kernel.InvokePromptAsync(prompt);
Console.WriteLine(answer.GetValue<string>());
*/