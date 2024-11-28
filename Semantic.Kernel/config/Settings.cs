// Copyright (c) Microsoft. All rights reserved.
using System.Text.Json;

public static class Settings
{
    private const string DefaultConfigFile = "../config/settings.json";
    private const string TypeKey = "type";
    private const string ModelKey = "model";
	private const string EmbeddingKey = "embedding";
    private const string EndpointKey = "endpoint";
    private const string ApiKey = "apikey";
    private const string OrgKey = "org";
    private const string BingKey = "bing";
    private const string TenantKey = "tenant";
    private const string ClientKey = "client";
    private const string SecretKey = "secret";
	private const string SearchApiKey = "searchService";
	private const string SearchEndpointKey = "searchKey";

    private const bool StoreConfigOnFile = true;

    // Load settings from file
    public static (bool useAzureOpenAI, string model, string azureEndpoint, string apiKey, string orgId)
        LoadFromFile(string configFile = DefaultConfigFile)
    {
        if (!File.Exists(configFile))
        {
            Console.WriteLine("Configuration not found: " + configFile);
            Console.WriteLine("\nPlease run the Setup Notebook (0-AI-settings.ipynb) to configure your AI backend first.\n");
            throw new Exception("Configuration not found, please setup the notebooks first using notebook 0-AI-settings.pynb");
        }

        try
        {
            var config = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(configFile));
            bool useAzureOpenAI = config[TypeKey] == "azure";
            string model = config[ModelKey];
            string azureEndpoint = config[EndpointKey];
            string apiKey = config[ApiKey];
            string orgId = config[OrgKey];
            if (orgId == "none") { orgId = ""; }

            return (useAzureOpenAI, model, azureEndpoint, apiKey, orgId);
        }
        catch (Exception e)
        {
            Console.WriteLine("Something went wrong: " + e.Message);
            return (true, "", "", "", "");
        }
    }

    // Read and return settings from file
    private static (bool useAzureOpenAI, string model, string azureEndpoint, string apiKey, string orgId)
        ReadSettings(bool _useAzureOpenAI, string configFile)
    {
        // Save the preference set in the notebook
        bool useAzureOpenAI = _useAzureOpenAI;
        string model = "";
        string azureEndpoint = "";
        string apiKey = "";
        string orgId = "";

        try
        {
            if (File.Exists(configFile))
            {
                (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = LoadFromFile(configFile);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Something went wrong: " + e.Message);
        }

        // If the preference in the notebook is different from the value on file, then reset
        if (useAzureOpenAI != _useAzureOpenAI)
        {
            useAzureOpenAI = _useAzureOpenAI;
            model = "";
            azureEndpoint = "";
            apiKey = "";
            orgId = "";
        }

        return (useAzureOpenAI, model, azureEndpoint, apiKey, orgId);
    }

    public static string BingSearchKey()
    {
        string configFile = DefaultConfigFile;

        var config = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(configFile));
        return config[BingKey];
    }

    public static string EmbeddingModelKey()
    {
        string configFile = DefaultConfigFile;

        var config = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(configFile));
        return config[EmbeddingKey];
    }
	
    public static (string searchEndpoint, string searchApikey) SearchSettings()
    {
        string configFile = DefaultConfigFile;

        var config = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(configFile));
        return (config[SearchEndpointKey], config[SearchApiKey]);
    }	
	
    public static (string tenant, string client, string secret) GraphSettings()
    {
        string configFile = DefaultConfigFile;

        var config = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(configFile));
        return (config[TenantKey], config[ClientKey], config[SecretKey]);
    }
}
