using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Numerics.Tensors;
using Microsoft.Extensions.AI;

IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator = new OllamaEmbeddingGenerator(
    new Uri("http://localhost:11434"),
    modelId: "all-minilm"
);

/*
var single_word = "coffee";
var result = await embeddingGenerator.GenerateEmbeddingAsync(single_word);
Console.WriteLine($"Vector of length for {single_word}: {result.Vector.Length}");
foreach (var value in result.Vector.Span)
{
    Console.Write("{0:0.00}, ", value);
}
*/

string filePath = "text_data.csv";
 // Read all lines from the file
string[] lines = File.ReadAllLines(filePath);

// Skip the header and trim the quotes
string[] word_list = lines
    .Skip(1) // Skip the header
    .Select(line => line.Trim('"')) // Remove double quotes
    .ToArray();

Console.WriteLine("\n\nGenerating embeddings for word list...");
var wordListEmbeddings = await embeddingGenerator.GenerateAndZipAsync(word_list);

/*
while (true)
{
    Console.WriteLine("\nQuery: ");
    var input = Console.ReadLine()!;
    if (input == "") break;

    var inputEmbedding = await embeddingGenerator.GenerateEmbeddingAsync(input);

    var closest =
        from word in wordListEmbeddings
        let similarity = TensorPrimitives.CosineSimilarity(word.Embedding.Vector.Span, inputEmbedding.Vector.Span)
        orderby similarity descending
        select new { Text = word.Value, Similarity = similarity };

    foreach (var c in closest.Take(3))
    {
        Console.WriteLine($"({c.Similarity}): {c.Text}");
    }
}
*/

// Hardcoded query: "king - man + woman"
Console.WriteLine("\nClosest words to 'king - man + woman':");
var kingEmbedding = await embeddingGenerator.GenerateEmbeddingAsync("king");
var manEmbedding = await embeddingGenerator.GenerateEmbeddingAsync("man");
var womanEmbedding = await embeddingGenerator.GenerateEmbeddingAsync("woman");

// Perform vector arithmetic: king - man + woman
float[] resultVector = new float[kingEmbedding.Vector.Length];
for (int i = 0; i < resultVector.Length; i++)
{
    resultVector[i] = kingEmbedding.Vector.Span[i] - manEmbedding.Vector.Span[i] + womanEmbedding.Vector.Span[i];
}

// Find the closest words to the resulting vector
var closest =
    from word in wordListEmbeddings
    let similarity = TensorPrimitives.CosineSimilarity(word.Embedding.Vector.Span, resultVector)
    orderby similarity descending
    select new { Text = word.Value, Similarity = similarity };

foreach (var c in closest.Take(3))
{
    Console.WriteLine($"({c.Similarity}): {c.Text}");
}