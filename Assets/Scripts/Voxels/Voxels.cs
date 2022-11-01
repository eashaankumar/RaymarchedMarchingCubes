using System;
using UnityEngine;

//[ExecuteInEditMode, ImageEffectAllowedInSceneView]
public class Voxels : MonoBehaviour
{

    [SerializeField]
    float scale;
    [SerializeField]
    float amplitude;
    [SerializeField]
    int voxelsPerAxis;
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
    Color waterColor;

    public ComputeShader voxelShader;

    RenderTexture target;
    Camera cam;
    Light directionalLight;

    ComputeBuffer voxelBuffer;
    int kernelIndex;

    uint[] voxels;

    void Start()
    {
        voxels = new uint[voxelsPerAxis * voxelsPerAxis * voxelsPerAxis];
        kernelIndex = voxelShader.FindKernel("CSMain");
    }

    void Init()
    {
        cam = Camera.current;
        directionalLight = FindObjectOfType<Light>();
    }

    int to1D(int x, int y, int z)
    {
        return (voxelsPerAxis * voxelsPerAxis * z) + (y * voxelsPerAxis) + x;
    }

    // Animate properties
    void Update()
    {
        // update grid
        voxels[0] = 1;
        /*if (Input.GetMouseButtonDown(0))
        {
            Vector3Int pos = new Vector3Int(Mathf.FloorToInt(transform.position.x),
                                            Mathf.FloorToInt(transform.position.y),
                                            Mathf.FloorToInt(transform.position.z));
            voxels[to1D(pos.x, pos.y, pos.z) % voxels.Length] = 1;
        }*/
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Init();
        InitRenderTexture();
        SetParameters();

        int threadGroupsX = Mathf.CeilToInt(cam.pixelWidth / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(cam.pixelHeight / 8.0f);
        voxelShader.Dispatch(0, threadGroupsX, threadGroupsY, 1);

        Graphics.Blit(target, destination, new Vector2(-1, -1), new Vector2(1, 1));
    }

    void SetParameters()
    {
        voxelShader.SetTexture(0, "Destination", target);
        voxelShader.SetFloats("scale", scale);
        voxelShader.SetFloats("amplitude", amplitude);
        voxelShader.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix);
        voxelShader.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse);
        voxelShader.SetVector("_LightDirection", directionalLight.transform.forward);
        voxelShader.SetInt("voxelsPerAxis", voxelsPerAxis);
        voxelShader.SetVector("_AmbientColor", new Vector3(ambientColor.r, ambientColor.g, ambientColor.b));
        voxelShader.SetVector("_GrassColor", new Vector3(grassColor.r, grassColor.g, grassColor.b));
        voxelShader.SetVector("_SandColor", new Vector3(sandColor.r, sandColor.g, sandColor.b));
        voxelShader.SetVector("_DirtColor", new Vector3(dirtColor.r, dirtColor.g, dirtColor.b));
        voxelShader.SetVector("_WaterColor", new Vector3(waterColor.r, waterColor.g, waterColor.b));


        voxelBuffer = new ComputeBuffer((int)(voxelsPerAxis *
                                        voxelsPerAxis *
                                        voxelsPerAxis),
                                        sizeof(uint));
        voxelBuffer.SetData(voxels);
        voxelShader.SetBuffer(kernelIndex, "voxels", voxelBuffer);

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
    }
}
