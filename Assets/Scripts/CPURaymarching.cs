using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class CPURaymarching : MonoBehaviour
{
    [Header("Rendering Engine")]
    [SerializeField, Tooltip("# of ticks between each render call (for updating destination render texture)")]
    int renderTicks;
    [SerializeField, Tooltip("Resolution"), Range(0f, 1f)]
    float resolution;

    int ticks;

    Texture2D tex;
    Camera cam;
    Light sun;

    int width, height;

    NativeArray<float4> pixels;

    void Start()
    {
        ticks = 0;

        cam = Camera.main;
        sun = FindObjectOfType<Light>();
        FillDensity();
    }

    private void OnDestroy()
    {
        pixels.Dispose();
    }

    void FillDensity()
    {
        /*int densityKernel = voxelShader.FindKernel("DensityKernel");

        Init();
        InitRenderTexture();
        SetParameters(densityKernel);

        int threadGroupsSize = Mathf.CeilToInt(1000 / 8.0f);
        voxelShader.Dispatch(densityKernel, threadGroupsSize, threadGroupsSize, threadGroupsSize);*/
    }

    private void Update()
    {
        ticks++;
        if (ticks > renderTicks)
        {
            ticks = 0;
            Render();
        }
    }

    void Init()
    {
        width = (int)(cam.pixelWidth * resolution);
        height = (int)(cam.pixelHeight * resolution);

        tex = new Texture2D(width, height, TextureFormat.RGBAFloat, false);
        tex.filterMode = FilterMode.Point;

        if (pixels != null && pixels.IsCreated) pixels.Dispose();
        pixels = new NativeArray<float4>(width * height, Allocator.Persistent);

    }

    private void Render()
    {
        Init();

        
        RenderJob job = new RenderJob()
        {
            pixels = pixels,
            width = width,
            height = height,
        };
        JobHandle handle = job.Schedule(pixels.Length, 64);
        handle.Complete();

        tex.SetPixelData<float4>(pixels, 0);
        tex.Apply();

    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(tex, destination);
    }

    #region Jobs
    struct RenderJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<float4> pixels;

        [ReadOnly] public int width;
        [ReadOnly] public int height;

        public void Execute(int index)
        {
            int2 uv = to2D(index);
            float2 uvNorm = new float2(uv.x / (float)width, uv.y / (float)height);
            pixels[index] = new float4(uvNorm, 0, 0);
        }

        int2 to2D(int index)
        {
            return new int2(index % width, index / width);
        }
    }

    #endregion
}
