using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
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
    [SerializeField]
    float maxRenderDist;

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
        tex.filterMode = FilterMode.Trilinear;

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
            _CameraToWorld = cam.cameraToWorldMatrix,
            _CameraInverseProjection = cam.projectionMatrix.inverse,
            maxDist = maxRenderDist,
            planetCenter = new float3(0, 0, 0),
            planetRadius = 1,
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
    struct Ray
    {
        public float3 origin;
        public float3 direction;
    }

    struct RaymarchResult
    {
        public bool hit;
        public float d;
    }

    [BurstCompile]
    struct RenderJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeArray<float4> pixels;

        [ReadOnly] public int width;
        [ReadOnly] public int height;
        [ReadOnly] public float4x4 _CameraToWorld;
        [ReadOnly] public float4x4 _CameraInverseProjection;
        [ReadOnly] public float maxDist;

        const float epsilon = 0.0001f;
        [ReadOnly] public float3 planetCenter;
        [ReadOnly] public float planetRadius;

        public void Execute(int index)
        {
            int2 uv = to2D(index);
            float2 uvNorm = new float2(uv.x / (float)width, uv.y / (float)height);

            Ray ray = CreateCameraRay(uvNorm * 2 - 1);

            RaymarchResult rs = raymarch(ray);

            
            pixels[index] = getColor(rs, ray);
        }

        float4 getColor(RaymarchResult res, Ray origin)
        {
            float4 color = new float4(0, 0, 0, 0);
            if (res.hit)
            {
                float3 hitPoint = origin.origin + origin.direction * res.d;
                float3 normal = math.normalize(hitPoint - planetCenter);
                color = new float4(normal.xyz * 0.5f + 0.5f, 1);
            }
            return color;
        }

        RaymarchResult raymarch(Ray start)
        {
            float d = 0;
            Ray current = start;
            RaymarchResult res = new RaymarchResult();
            while (d < maxDist)
            {
                float3 p = current.origin + current.direction * d;
                float sdf = sdfSphere(p, planetCenter, planetRadius);
                d += sdf;
                if (sdf < epsilon)
                {
                    res.hit = true;
                    res.d = d;
                    return res;
                }
            }
            res.hit = false;
            return res;
        }

        float sdfSphere(float3 p, float3 center, float radius)
        {
            return math.distance(p, center) - radius;
        }

        int2 to2D(int index)
        {
            return new int2(index % width, index / width);
        }

        Ray CreateRay(float3 origin, float3 direction)
        {
            Ray ray = new Ray();
            ray.origin = origin;
            ray.direction = direction;
            return ray;
        }

        Ray CreateCameraRay(float2 uv)
        {
            float3 origin = math.mul(_CameraToWorld, new float4(0, 0, 0, 1)).xyz;
            float3 direction = math.mul(_CameraInverseProjection, new float4(uv, 0, 1)).xyz;
            direction = math.mul(_CameraToWorld, new float4(direction, 0)).xyz;
            direction = math.normalize(direction);
            return CreateRay(origin, direction);
        }
    }

    #endregion
}
