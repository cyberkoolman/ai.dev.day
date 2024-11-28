using Azure.Identity;
using Microsoft.Graph;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Plugins.MsGraph;
using Microsoft.SemanticKernel.Plugins.MsGraph.Connectors;
using Kernel = Microsoft.SemanticKernel.Kernel;

var builder = Kernel.CreateBuilder();

// Configure AI backend used by the kernel
var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

if (useAzureOpenAI)
    builder.AddAzureOpenAIChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOpenAIChatCompletion(model, apiKey, orgId);

// Build the kernel
var kernel = builder.Build();

// The client credentials flow requires that you request the
// /.default scope, and pre-configure your permissions on the
// app registration in Azure. An administrator must grant consent
// to those permissions beforehand.
var scopes = new[] { "Calendars.Read" };

// Multi-tenant apps can use "common",
// single-tenant apps must use the tenant ID from the Azure portal
var (tenantId, clientId, clientSecret) = Settings.GraphSettings();
var options = new DeviceCodeCredentialOptions
{
    AuthorityHost = AzureAuthorityHosts.AzurePublicCloud,
    ClientId = clientId,
    TenantId = tenantId,

    // Callback function that receives the user prompt
    // Prompt contains the generated device code that user must
    // enter during the auth process in the browser
    DeviceCodeCallback = (code, cancellation) =>
    {
        Console.WriteLine(code.Message);
        return Task.FromResult(0);
    },
};

// https://learn.microsoft.com/dotnet/api/azure.identity.devicecodecredential
var deviceCodeCredential = new DeviceCodeCredential(options);

var graphClient = new GraphServiceClient(deviceCodeCredential, scopes);
await graphClient.Me.Request().GetAsync();

OutlookCalendarConnector connector = new OutlookCalendarConnector(graphClient);

CalendarPlugin calendarPlugin = new CalendarPlugin(connector);

// Just to verify manually that the plugin works
// var events = await calendarPlugin.GetCalendarEventsAsync(10, 0, CancellationToken.None);

kernel.ImportPluginFromObject(calendarPlugin, "CalendarPlugin");

OpenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
};

string prompt = "What is my next meeting?";
Console.WriteLine("=== Your Next Meeting ===");
var results = kernel.InvokePromptStreamingAsync(prompt, new KernelArguments(settings));
// var results = await kernel.InvokePromptAsync(prompt);

await foreach (var message in results)
{
    Console.Write(message);
}
