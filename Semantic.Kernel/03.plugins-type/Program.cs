using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Plugins.Core;

// Configure AI service credentials used by the kernel
var (useAzureOpenAI, chatDeployment, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(chatDeployment, azureEndpoint, apiKey)
    .Build();
var question = "what is 12.34 * 34.56?";
Console.WriteLine(question);

// Option 1: Use LLM
// var result = await kernel.InvokePromptAsync(question);

// Option 2: Use SK Provided Math Plugin
// https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/Plugins/Plugins.Core/MathPlugin.cs

// kernel.Plugins.AddFromType<MathPlugin>();
// PromptExecutionSettings settings = new()
// {
//     FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
// };
// var result = await kernel.InvokePromptAsync(question, new (settings));

 
// Option 3: Use Custom Math Plugin
kernel.Plugins.AddFromType<CustomPlugins.MathPlugin>();
PromptExecutionSettings settings = new()
{
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
};
var result = await kernel.InvokePromptAsync(question, new KernelArguments(settings));

// Print the result
Console.WriteLine(result);
