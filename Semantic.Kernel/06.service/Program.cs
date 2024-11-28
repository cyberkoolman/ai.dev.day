using System.Text;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

var (useAzureOpenAI, chatDeployment, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(chatDeployment, azureEndpoint, apiKey)
    .Build();

kernel.ImportPluginFromFunctions("HelperFunctions", new[]
{
    kernel.CreateFunctionFromMethod((string cityName) =>
        cityName switch
        {
            "Boston" => "61 and rainy",
            "London" => "55 and cloudy",
            "Miami" => "80 and sunny",
            "Paris" => "60 and rainy",
            "Tokyo" => "50 and sunny",
            "Sydney" => "75 and sunny",
            "Tel Aviv" => "80 and sunny",
            _ => "31 and snowing",
        }, "Get_Weather_For_City", "Gets the current weather for the specified city"),
});

PromptExecutionSettings settings = new() 
{
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() 
};
var chat = kernel.GetRequiredService<IChatCompletionService>();
var chatHistory = new ChatHistory();

while (true)
{
    Console.Write("Question (Type \"quit\" to leave): ");
    string question = Console.ReadLine() ?? string.Empty;
    if (question == "quit")
    {
        break;
    }

    chatHistory.AddUserMessage(question);
    StringBuilder sb = new();
    await foreach (var update in chat.GetStreamingChatMessageContentsAsync(chatHistory, settings, kernel))
    {
        if (update.Content is not null)
        {
            Console.Write(update.Content);
            sb.Append(update.Content);
        }
    }
    chatHistory.AddAssistantMessage(sb.ToString());
    Console.WriteLine();
}
