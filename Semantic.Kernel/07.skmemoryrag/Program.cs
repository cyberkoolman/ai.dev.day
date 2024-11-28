using System.Text.Json;
using KernelMemory.Models;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

// Common Setting: Get Settings from Settings.json
var (useAzureOpenAI, chatDeployment, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
var question = "What is Contoso Electronics?";

// Section 1: Testing with Azure OpenAI as is
// var kernel = Kernel.CreateBuilder()
//     .AddAzureOpenAIChatCompletion(chatDeployment, azureEndpoint, apiKey)
//     .Build();
// var answer = await kernel.InvokePromptAsync(question);
// Console.WriteLine(answer.GetValue<string>());


// Section 2: Create a Semantic Kernel with local memory for RAG
var embbeddingDeplpoyment = Settings.EmbeddingModelKey();

var embeddingConfig = new AzureOpenAIConfig
{
    APIKey = apiKey,
    Deployment = embbeddingDeplpoyment,
    Endpoint = azureEndpoint,
    APIType = AzureOpenAIConfig.APITypes.EmbeddingGeneration,
    Auth = AzureOpenAIConfig.AuthTypes.APIKey
};

var chatConfig = new AzureOpenAIConfig
{
    APIKey = apiKey,
    Deployment = chatDeployment,
    Endpoint = azureEndpoint,
    APIType = AzureOpenAIConfig.APITypes.ChatCompletion,
    Auth = AzureOpenAIConfig.AuthTypes.APIKey
};

var kernelMemory = new KernelMemoryBuilder()
    .WithAzureOpenAITextGeneration(chatConfig)
    .WithAzureOpenAITextEmbeddingGeneration(embeddingConfig)
    .WithSimpleVectorDb()
    .Build<MemoryServerless>();

// Import documents
var filename = "employee_handbook.pdf";
var employeeHandbook = Path.Combine(System.IO.Directory.GetCurrentDirectory(), "Documents", filename);
await kernelMemory.ImportDocumentAsync(employeeHandbook);

var answer = await kernelMemory.AskAsync(question);
var response = new KernelResponse
{
    Answer = answer.Result,
    Citations = answer.RelevantSources
};
Console.WriteLine(response.Answer);

// var citation = JsonSerializer.Serialize(response.Citations.FirstOrDefault());
// Console.WriteLine($"Citation: {citation}");

var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(chatDeployment, azureEndpoint, apiKey)
    .Build();

var pluginsDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Plugins");
kernel.ImportPluginFromPromptDirectory(pluginsDirectory + "\\MailPlugin", "MailPlugin");

var plugin = new MemoryPlugin(kernelMemory, waitForIngestionToComplete: true);
kernel.ImportPluginFromObject(plugin, "memory");

// RAG combined with plugins
OpenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions,
};

var chatHistory = new ChatHistory();
var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

while (true)
{
   Console.Write("User > ");
   question = Console.ReadLine();

   var prompt = $@"
           Question to Kernel Memory: {question}
           Kernel Memory Answer: {{memory.ask}}
           If the answer is empty say 'I don't know', otherwise reply with the answer.
           ";

   chatHistory.AddMessage(AuthorRole.User, prompt);
   var result = await chatCompletionService.GetChatMessageContentAsync(chatHistory, settings, kernel);
   Console.WriteLine($"Assistant: {result.Content}");
   chatHistory.AddMessage(AuthorRole.Assistant, result.Content);
}