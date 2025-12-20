# Use the official .NET SDK image to build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy solution and project files
COPY Larrak.CEM.sln ./
COPY src/Larrak.CEM.API/Larrak.CEM.API.csproj src/Larrak.CEM.API/
COPY src/Larrak.CEM.Core/Larrak.CEM.Core.csproj src/Larrak.CEM.Core/
COPY src/Larrak.CEM.Engine/Larrak.CEM.Engine.csproj src/Larrak.CEM.Engine/

# Restore dependencies
RUN dotnet restore

# Copy the rest of the source code
COPY . .

# Build and publish
WORKDIR /src/src/Larrak.CEM.API
RUN dotnet publish -c Release -o /app/publish

# Use the runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 50051
ENTRYPOINT ["dotnet", "Larrak.CEM.API.dll"]
