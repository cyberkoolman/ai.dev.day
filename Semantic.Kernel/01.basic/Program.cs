using Microsoft.SemanticKernel;

// Get Settings from Settings.json
var (useAzureOpenAI, chatDeployment, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

var kernel = useAzureOpenAI ? Kernel.CreateBuilder()
                                .AddAzureOpenAIChatCompletion(chatDeployment, azureEndpoint, apiKey)
                                .Build()
                            : Kernel.CreateBuilder()
                                .AddOpenAIChatCompletion("gpt-3.5-turbo", apiKey, orgId)
                                .Build();
var question = "What is Contoso Electronics?";

// Case 1. Just Ask a question, and the question is the prompt
// var answer = await kernel.InvokePromptAsync(question);
// Console.WriteLine(answer.GetValue<string>());

// Case 2: Define and Configure Inline Prompt and a sample input text
var sourceData = """
Contoso Electronics, a global leader in innovative technology, began as a small family-run business in the 1970s,
repairing radios and televisions in a bustling urban neighborhood. Over the decades, it evolved into a multi-billion-dollar
enterprise renowned for its cutting-edge consumer electronics and IoT solutions. Headquartered in Redmond, Washington,
Contoso prides itself on designing products that seamlessly integrate technology into everyday life.

The company’s flagship product line, the Contoso SmartHub, revolutionized home automation with intuitive
AI-powered voice control and energy-saving features, earning accolades worldwide. Contoso expanded its portfolio
to include electric vehicles, wearable tech, and enterprise solutions for businesses seeking scalable IoT infrastructures.

Guided by a mission to "Innovate for a Better Tomorrow," Contoso champions sustainability, incorporating
recycled materials into its products and implementing zero-waste manufacturing practices.
With a diverse team of over 50,000 employees in 30 countries, Contoso Electronics is a beacon of innovation,
customer-centricity, and environmental stewardship.
""";

string promptString = """
{{$content}}

{{$question}}
""";

var summaryFunction = kernel.CreateFunctionFromPrompt(promptString);
var summaryResult = await kernel.InvokeAsync(summaryFunction, new() { ["content"] = sourceData, ["question"] = question });
Console.WriteLine(summaryResult);

// Case 3: Chat with the Assistant
const string chatPrompt = @"
{{$history}}
User: {{$input}}
Assistant:";

var chatFunction = kernel.CreateFunctionFromPrompt(chatPrompt);
var history = $"\nUser: {question}\nAssistant: {summaryResult}\n";
var arguments = new KernelArguments()
{
    ["history"] = history
};

Func<string, Task> Chat = async (string input) => {
    arguments["input"] = input;
    var answer = await chatFunction.InvokeAsync(kernel, arguments);

    // Append the new interaction to the chat history
    var result = $"\nUser: {input}\nAssistant: {answer}\n";
    history += result;
    arguments["history"] = history;
    
    Console.WriteLine(result);
};

await Chat("Can you find anything about their products?");
await Chat("Tell me more about their first product you just mentioned");
