using System.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using UnityEngine;

public class MarchingCubesRayMarch : MonoBehaviour
{
    [SerializeField]
    float scale;
    [SerializeField]
    float impact;
    [SerializeField]
    float octaves;
    [SerializeField, Tooltip("Compensation for adding noise")]
    float planetRadiusOffset;
    [Header("Generator settings")]
    [SerializeField]
    Texture2D sandTexture;
    [SerializeField]
    Texture2D waterTexture;
    [SerializeField]
    Color ambientColor;
    [SerializeField]
    Color grassColor;
    [SerializeField]
    Color dirtColor;
    [SerializeField]
    Color sandColor;
    [SerializeField]
    Color waterColorShallow;
    [SerializeField]
    Color waterColorDeep;
    [SerializeField]
    float brushStrength;
    [SerializeField]
    int brushSize;

    [Header("Planet Settings")]
    [SerializeField]
    float planetRadius;

    [Header("Ocean Settings")]
    [SerializeField]
    float oceanRadius;
    [SerializeField, Tooltip("Depth at which waves form on the shores")]
    float waveBreakDepth;
    [SerializeField]
    float waterDensityFallOff;

    [Header("Atmosphere Settings")]
    [SerializeField]
    float atmosphereRadius;
    [SerializeField]
    float atmosphereDensityFallOff;
    [SerializeField]
    int opticalDepthSteps;
    [SerializeField]
    int atmosphereSteps;
    [SerializeField]
    Vector3 wavelengths = new Vector3(700, 530, 440);
    [SerializeField]
    float scatteringStrength;
    [SerializeField]
    float specularStrength;

    [Header("Rendering Engine")]
    [SerializeField, Tooltip("# of ticks between each render call (for updating destination render texture)")]
    int renderTicks;
    [SerializeField, Tooltip("Resolution"), Range(0f, 1f)]
    float resolution;

    public ComputeShader voxelShader;

    int ticks;

    ComputeBuffer mapPosCenterBuffer;
    Vector3Int[] mapPosCenter;

    RenderTexture target, density, waterDensity;
    Camera cam;
    Light sun;

    int kernelIndex;
    int width, height;
    bool renderOdds;

    void Start()
    {
        kernelIndex = voxelShader.FindKernel("CSMain");

        mapPosCenterBuffer = new ComputeBuffer(1, sizeof(int) * 3);
        mapPosCenter = new Vector3Int[1];
        ticks = 0;

        cam = Camera.main;
        sun = FindObjectOfType<Light>();
        FillDensity();
    }

    void FillDensity()
    {
        int densityKernel = voxelShader.FindKernel("DensityKernel");

        Init();
        InitRenderTexture();
        SetParameters(densityKernel);

        int threadGroupsSize = Mathf.CeilToInt(1000 / 8.0f);
        voxelShader.Dispatch(densityKernel, threadGroupsSize, threadGroupsSize, threadGroupsSize);
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
    }

    private void Render()
    {
        Init();
        InitRenderTexture();
        SetParameters(kernelIndex);

        int threadGroupsX = Mathf.CeilToInt(width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(height / 8.0f);
        voxelShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);
        GetData();
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(target, destination);
    }

    void GetData()
    {
        mapPosCenterBuffer.GetData(mapPosCenter);
    }

    void SetParameters(int kernelIndex)
    {
        brushStrength = Mathf.Abs(brushStrength);
        voxelShader.SetTexture(kernelIndex, "Destination", target);
        voxelShader.SetTexture(kernelIndex, "Density", density);
        voxelShader.SetTexture(kernelIndex, "WaterDensity", waterDensity);
        voxelShader.SetTexture(kernelIndex, "SandTexture", sandTexture);
        voxelShader.SetTexture(kernelIndex, "WaterTexture", waterTexture);
        voxelShader.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix);
        voxelShader.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse);
        voxelShader.SetVector("_LightDirection", sun.transform.forward);
        voxelShader.SetVector("_AmbientColor", new Vector3(ambientColor.r, ambientColor.g, ambientColor.b));
        voxelShader.SetVector("_GrassColor", new Vector3(grassColor.r, grassColor.g, grassColor.b));
        voxelShader.SetVector("_SandColor", new Vector3(sandColor.r, sandColor.g, sandColor.b));
        voxelShader.SetVector("_DirtColor", new Vector3(dirtColor.r, dirtColor.g, dirtColor.b));
        voxelShader.SetVector("_WaterColorShallow", new Vector3(waterColorShallow.r, waterColorShallow.g, waterColorShallow.b));
        voxelShader.SetVector("_WaterColorDeep", new Vector3(waterColorDeep.r, waterColorDeep.g, waterColorDeep.b));
        voxelShader.SetFloat("_Scale", scale);
        voxelShader.SetFloat("_Impact", impact);
        voxelShader.SetFloat("_Octaves", octaves);
        voxelShader.SetFloat("_Time", Time.time);
        voxelShader.SetFloat("_WaveBreakDepth", waveBreakDepth);
        voxelShader.SetFloat("_SpecularStrength", specularStrength);
        voxelShader.SetVector("_SunColor", new Vector3(sun.color.r, sun.color.g, sun.color.b));
        voxelShader.SetFloat("planetRadius", planetRadiusOffset + planetRadius);

        #region Atmosphere shader variables
        voxelShader.SetFloat("_AtmosphereRadius", atmosphereRadius);
        voxelShader.SetFloat("_OceanRadius", oceanRadius);
        voxelShader.SetFloat("_AtmosphereDensityFalloff", atmosphereDensityFallOff);
        voxelShader.SetInt("_OpticalDepthSteps", opticalDepthSteps);
        voxelShader.SetInt("_AtmosphereSteps", atmosphereSteps);
        voxelShader.SetFloat("_WaterDensityFalloff", waterDensityFallOff);
        float scatterR = Mathf.Pow(400 / wavelengths.x, 4) * scatteringStrength;
        float scatterG = Mathf.Pow(400 / wavelengths.y, 4) * scatteringStrength;
        float scatterB = Mathf.Pow(400 / wavelengths.z, 4) * scatteringStrength;

        voxelShader.SetVector("_ScatterCoefficients", new Vector3(scatterR, scatterG, scatterB));
        #endregion
        int terra = 0;
        if (Input.GetMouseButton(0)) terra = 1;
        else if (Input.GetMouseButton(1)) terra = -1;
        voxelShader.SetInt("_Terraforming", terra);
        voxelShader.SetFloat("_BrushStrength", brushStrength);
        voxelShader.SetInt("_BrushSize", brushSize);

        mapPosCenterBuffer.SetData(mapPosCenter);
        voxelShader.SetBuffer(kernelIndex, "mapPosCenter", mapPosCenterBuffer);

        #region Rendering variables
        renderOdds = !renderOdds;
        voxelShader.SetBool("_RenderOdds", renderOdds);
        #endregion

    }

    void InitRenderTexture()
    {
        if (target == null || target.width != width || target.height != height)
        {
            if (target != null)
            {
                target.Release();
            }
            target = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            target.enableRandomWrite = true;
            target.Create();
        }
        Create3DTexture(ref density, 1000, "Density");
        Create3DTexture(ref waterDensity, 1000, "WaterDensity");
    }

    void Create3DTexture(ref RenderTexture texture, int size, string name)
    {
        //
        var format = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat;
        if (texture == null || !texture.IsCreated() || texture.width != size || texture.height != size || texture.volumeDepth != size || texture.graphicsFormat != format)
        {
            //Debug.Log ("Create tex: update noise: " + updateNoise);
            if (texture != null)
            {
                texture.Release();
            }
            const int numBitsInDepthBuffer = 0;
            texture = new RenderTexture(size, size, numBitsInDepthBuffer);
            texture.graphicsFormat = format;
            texture.volumeDepth = size;
            texture.enableRandomWrite = true;
            texture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;


            texture.Create();

            texture.wrapMode = TextureWrapMode.Repeat;
            texture.filterMode = FilterMode.Bilinear;
            texture.name = name;
        }
    }

    #region Jobs
    struct RenderJob : IJobParallelFor
    {
        public void Execute(int index)
        {
            
        }
    }

    #endregion
}
