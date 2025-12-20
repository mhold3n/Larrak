using Larrak.CEM.API.Services;

var builder = WebApplication.CreateBuilder(args);

// Add gRPC services
builder.Services.AddGrpc();

// Configure port from environment variable or default
var port = Environment.GetEnvironmentVariable("CEM_PORT") ?? "50051";
builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenLocalhost(int.Parse(port), listenOptions =>
    {
        listenOptions.Protocols = Microsoft.AspNetCore.Server.Kestrel.Core.HttpProtocols.Http2;
    });
});

var app = builder.Build();

// Map gRPC service
app.MapGrpcService<CEMServiceImpl>();

// Health check endpoint (for non-gRPC clients)
app.MapGet("/health", () => "OK");

Console.WriteLine($"CEM gRPC service starting on port {port}...");
Console.WriteLine($"Platform: {Environment.OSVersion.Platform}");
Console.WriteLine($"Runtime: {Environment.Version}");

app.Run();
