using Microsoft.SemanticKernel;

// Configure AI service credentials used by the kernel
var (useAzureOpenAI, chatDeployment, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(chatDeployment, azureEndpoint, apiKey)
    .Build();

var funPluginDirectoryPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "plugins", "FunPlugin");
var funPluginFunctions = kernel.ImportPluginFromPromptDirectory(funPluginDirectoryPath);
var skFunction = funPluginFunctions["Joke"];
var skArguments = new KernelArguments { ["input"] = "time travel to dinosaur age" };

var result = await kernel.InvokeAsync(skFunction, skArguments);
Console.WriteLine(result);