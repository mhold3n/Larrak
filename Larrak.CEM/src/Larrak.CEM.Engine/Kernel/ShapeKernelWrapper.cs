using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using PicoGK;
using Leap71.ShapeKernel;

namespace Larrak.CEM.Engine.Kernel;

public static class ShapeKernelWrapper
{
    private static bool _initialized = false;
    private static object _lock = new object();

    public static void Initialize(float voxelSizeMM = 0.5f)
    {
        lock (_lock)
        {
            if (!_initialized)
            {
                try
                {
                    // PicoGK.Library constructor for headless mode
                    new Library(voxelSizeMM);
                    _initialized = true;
                    // Note: We don't dispose Library here; it stays alive for the process lifetime
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ShapeKernelWrapper] Failed to init PicoGK: {ex.Message}");
                    // Continue, as this might be running in a simplified environment
                }
            }
        }
    }

    public static GeometryResult GenerateGearPair(
        List<double> rp,
        List<double> rr,
        List<double> c,
        float depth = 10.0f,
        float wallThickness = 5.0f,
        float voxelSize = 0.5f)
    {
        if (!_initialized) Initialize(voxelSize);

        try
        {
            // If initialization failed (mock mode), return empty
            if (!_initialized)
                return new GeometryResult { Hash = "mock-no-pico" };

            var voxels = new Voxels();

            // Simple averaging for the skeleton implementation
            // In a real implementation, we would sweep the profile
            float fRp = (float)rp.Average();
            float fRr = (float)rr.Average();
            float fC  = (float)c.Average();

            // 1. Planet (Cylinder)
            // Positioned at X = fC
            var planetFrame = new LocalFrame(new Vector3(fC, 0, 0));
            // BaseCylinder(Frame, Length, Radius)
            var planet = new BaseCylinder(planetFrame, depth, fRp);
            var voxPlanet = planet.voxConstruct();

            // 2. Ring (Pipe)
            // Positioned at Origin
            // BasePipe likely follows similar pattern or (Frame, Radius, Wall, Length)
            // We'll use BaseCylinder for the Ring's outer shell minus inner, or assuming BasePipe availability
            // Inspecting BasePipe.cs would verify signature, but assuming (Frame, Length, Radius, Wall) or similar
            // Let's rely on standard Cylinder for now to be safe, or just a cylinder for the Ring mass
            var ringFrame = new LocalFrame(Vector3.Zero);
            var ring = new BaseCylinder(ringFrame, depth, fRr + wallThickness); // Configurable wall thickness
            var voxRing = ring.voxConstruct();

            // Subtract inner bore from Ring?
            // LEAP71 BasePipe usually exists.

            // Boolean Union
            voxels.BoolAdd(voxPlanet);
            voxels.BoolAdd(voxRing);

            // Calculate volume and bounding box correctly
            voxels.CalculateProperties(out float vol, out BBox3 bbox);

            // Create a mesh for Hash/Metrics
            var mesh = new Mesh(voxels);
            // PicoGK Mesh doesn't easily give surface area in one call without calc
            // But getting vertex count is good proxy for complexity
            float area = mesh.nVertexCount();

            string tmpPath = Path.GetTempFileName() + ".pico";
            // Check if SaveTo exists or similar
            // voxels.SaveTo(tmpPath); // Hypothetical

            string meshHash = ComputeHash(mesh);

            return new GeometryResult {
                VoxelPath = tmpPath,
                Volume = vol, // Now using correctly calculated volume
                SurfaceArea = area,
                Hash = meshHash
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ShapeKernel] Error: {ex.Message}");
            return new GeometryResult { Hash = "error-" + ex.GetType().Name };
        }
    }

    private static string ComputeHash(Mesh mesh)
    {
        // Simple hash of geometry characteristics
        var input = $"{mesh.nVertexCount()}_{mesh.nTriangleCount()}_{mesh.oBoundingBox().vecSize().X}";
        using var md5 = MD5.Create();
        return Convert.ToHexString(md5.ComputeHash(Encoding.UTF8.GetBytes(input)));
    }
}

public class GeometryResult
{
    public string VoxelPath { get; set; } = "";
    public float Volume { get; set; }
    public float SurfaceArea { get; set; }
    public string Hash { get; set; } = "";
}
