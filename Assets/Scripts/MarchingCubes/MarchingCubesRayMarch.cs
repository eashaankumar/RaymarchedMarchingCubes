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
    [Header("Generator settings")]
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

    public ComputeShader voxelShader;

    ComputeBuffer mapPosCenterBuffer;
    Vector3Int[] mapPosCenter;

    RenderTexture target, density;
    Camera cam;
    Light directionalLight;

    int kernelIndex;

    void Start()
    {
        kernelIndex = voxelShader.FindKernel("CSMain");

        mapPosCenterBuffer = new ComputeBuffer(1, sizeof(int) * 3);
        mapPosCenter = new Vector3Int[1];
    }

    void Init()
    {
        cam = Camera.current;
        directionalLight = FindObjectOfType<Light>();


    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Init();
        InitRenderTexture();
        SetParameters();

        int threadGroupsX = Mathf.CeilToInt(cam.pixelWidth / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(cam.pixelHeight / 8.0f);
        voxelShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);
        GetData();
        

        Graphics.Blit(target, destination);
    }

    void GetData()
    {
        mapPosCenterBuffer.GetData(mapPosCenter);
    }

    void SetParameters()
    {
        brushStrength = Mathf.Abs(brushStrength);
        voxelShader.SetTexture(0, "Destination", target);
        voxelShader.SetTexture(0, "Density", density);
        voxelShader.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix);
        voxelShader.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse);
        voxelShader.SetVector("_LightDirection", directionalLight.transform.forward);
        voxelShader.SetVector("_AmbientColor", new Vector3(ambientColor.r, ambientColor.g, ambientColor.b));
        voxelShader.SetVector("_GrassColor", new Vector3(grassColor.r, grassColor.g, grassColor.b));
        voxelShader.SetVector("_SandColor", new Vector3(sandColor.r, sandColor.g, sandColor.b));
        voxelShader.SetVector("_DirtColor", new Vector3(dirtColor.r, dirtColor.g, dirtColor.b));
        voxelShader.SetVector("_WaterColorShallow", new Vector3(waterColorShallow.r, waterColorShallow.g, waterColorShallow.b));
        voxelShader.SetVector("_WaterColorDeep", new Vector3(waterColorDeep.r, waterColorDeep.g, waterColorDeep.b));
        voxelShader.SetFloat("_Scale", scale);
        voxelShader.SetFloat("_Impact", impact);
        int terra = 0;
        if (Input.GetMouseButton(0)) terra = 1;
        else if (Input.GetMouseButton(1)) terra = -1;
        voxelShader.SetInt("_Terraforming", terra);
        voxelShader.SetFloat("_BrushStrength", brushStrength);
        voxelShader.SetInt("_BrushSize", brushSize);

        mapPosCenterBuffer.SetData(mapPosCenter);
        voxelShader.SetBuffer(0, "mapPosCenter", mapPosCenterBuffer);
    }

    void InitRenderTexture()
    {
        if (target == null || target.width != cam.pixelWidth || target.height != cam.pixelHeight)
        {
            if (target != null)
            {
                target.Release();
            }
            target = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            target.enableRandomWrite = true;
            target.Create();
        }
        Create3DTexture(ref density, 1000, "Density");
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
        }
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Bilinear;
        texture.name = name;
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
