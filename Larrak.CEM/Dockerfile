# Use the official .NET SDK image to build
# Use the official .NET SDK image to build
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy solution and project files
COPY Larrak.CEM/Larrak.CEM.sln ./Larrak.CEM/
COPY Larrak.CEM/src/Larrak.CEM.API/Larrak.CEM.API.csproj ./Larrak.CEM/src/Larrak.CEM.API/
COPY Larrak.CEM/src/Larrak.CEM.Core/Larrak.CEM.Core.csproj ./Larrak.CEM/src/Larrak.CEM.Core/
COPY Larrak.CEM/src/Larrak.CEM.Engine/Larrak.CEM.Engine.csproj ./Larrak.CEM/src/Larrak.CEM.Engine/

# Copy dependencies
COPY PicoGK/PicoGK.csproj ./PicoGK/
COPY LEAP71_ShapeKernel/LEAP71_ShapeKernel.csproj ./LEAP71_ShapeKernel/

# Restore dependencies
RUN dotnet restore Larrak.CEM/Larrak.CEM.sln

# Copy the rest of the source code
COPY Larrak.CEM ./Larrak.CEM
COPY PicoGK ./PicoGK
COPY LEAP71_ShapeKernel ./LEAP71_ShapeKernel

# Build and publish
WORKDIR /src/Larrak.CEM/src/Larrak.CEM.API
RUN dotnet publish -c Release -o /app/publish

# Use the runtime image
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 50051
ENTRYPOINT ["dotnet", "Larrak.CEM.API.dll"]
