using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class MarchingCubesRayMarch : MonoBehaviour
{
    [SerializeField]
    float scale;
    [SerializeField]
    float amplitude;
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

    int kernelIndex;

    void Start()
    {
        kernelIndex = voxelShader.FindKernel("CSMain");
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

        Graphics.Blit(target, destination, new Vector2(-1, -1), new Vector2(1, 1));
    }

    void SetParameters()
    {
        voxelShader.SetTexture(0, "Destination", target);
        voxelShader.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix);
        voxelShader.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse);
        voxelShader.SetVector("_LightDirection", directionalLight.transform.forward);
        voxelShader.SetVector("_AmbientColor", new Vector3(ambientColor.r, ambientColor.g, ambientColor.b));
        voxelShader.SetVector("_GrassColor", new Vector3(grassColor.r, grassColor.g, grassColor.b));
        voxelShader.SetVector("_SandColor", new Vector3(sandColor.r, sandColor.g, sandColor.b));
        voxelShader.SetVector("_DirtColor", new Vector3(dirtColor.r, dirtColor.g, dirtColor.b));
        voxelShader.SetVector("_WaterColor", new Vector3(waterColor.r, waterColor.g, waterColor.b));

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
