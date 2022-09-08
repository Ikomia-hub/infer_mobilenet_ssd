#include "MobileNetSSD.h"
#include "IO/CObjectDetectionIO.h"

CMobileNetSSD::CMobileNetSSD() : COcvDnnProcess()
{
    m_pParam = std::make_shared<CMobileNetSSDParam>();
    addOutput(std::make_shared<CObjectDetectionIO>());
}

CMobileNetSSD::CMobileNetSSD(const std::string &name, const std::shared_ptr<CMobileNetSSDParam> &pParam): COcvDnnProcess(name)
{
    m_pParam = std::make_shared<CMobileNetSSDParam>(*pParam);
    addOutput(std::make_shared<CObjectDetectionIO>());
}

size_t CMobileNetSSD::getProgressSteps()
{
    return 3;
}

int CMobileNetSSD::getNetworkInputSize() const
{
    int size = 416;

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    auto pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(m_pParam);
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
        size = size + (m_sign * 32);

    return size;
}

double CMobileNetSSD::getNetworkInputScaleFactor() const
{
    return 1.0 / 127.5;
}

cv::Scalar CMobileNetSSD::getNetworkInputMean() const
{
    return 127.5;
}

void CMobileNetSSD::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(m_pParam);

    if(pInput == nullptr || pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    //Force model files path
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString();
    pParam->m_structureFile = pluginDir + "/Model/mobileNetSSD.prototxt";
    pParam->m_modelFile = pluginDir + "/Model/mobileNetSSD.caffemodel";
    pParam->m_labelsFile = pluginDir + "/Model/pascalVoc0712_names.txt";

    CMat imgSrc;
    CMat imgOrigin = pInput->getImage();
    std::vector<cv::Mat> netOutputs;

    //Detection networks need color image as input
    if(imgOrigin.channels() < 3)
        cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
    else
        imgSrc = imgOrigin;

    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn();
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;
            readClassNames();
            generateColors();
        }
        forward(imgSrc, netOutputs);
    }
    catch(cv::Exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    endTaskRun();
    emit m_signalHandler->doProgress();
    manageOutput(netOutputs[0]);
    emit m_signalHandler->doProgress();
}

void CMobileNetSSD::manageOutput(cv::Mat &dnnOutput)
{
    forwardInputImage();

    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    auto pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

    auto objDetectIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    objDetectIOPtr->init(getName(), 0);

    for(int i=0; i<dnnOutput.size[2]; i++)
    {
        //Detected class
        int classIndex[4] = { 0, 0, i, 1 };
        size_t classId = (size_t)dnnOutput.at<float>(classIndex);
        //Confidence
        int confidenceIndex[4] = { 0, 0, i, 2 };
        float confidence = dnnOutput.at<float>(confidenceIndex);

        if(confidence > pParam->m_confidence)
        {
            //Bounding box
            int leftIndex[4] = { 0, 0, i, 3 };
            int topIndex[4] = { 0, 0, i, 4 };
            int rightIndex[4] = { 0, 0, i, 5 };
            int bottomIndex[4] = { 0, 0, i, 6 };
            float left = dnnOutput.at<float>(leftIndex) * imgSrc.cols;
            float top = dnnOutput.at<float>(topIndex) * imgSrc.rows;
            float right = dnnOutput.at<float>(rightIndex) * imgSrc.cols;
            float bottom = dnnOutput.at<float>(bottomIndex) * imgSrc.rows;
            float width = right - left + 1;
            float height = bottom - top + 1;

            //Retrieve class label
            std::string className = classId < m_classNames.size() ? m_classNames[classId] : "unknown " + std::to_string(classId);

            objDetectIOPtr->addObject(i, className, confidence, left, top, width, height, m_colors[classId]);
        }
    }
}

void CMobileNetSSD::generateColors()
{
    //Random colors
    for(size_t i=0; i<m_classNames.size(); ++i)
    {
        m_colors.push_back({ (uchar)((double)std::rand() / (double)RAND_MAX * 255.0),
                             (uchar)((double)std::rand() / (double)RAND_MAX * 255.0),
                             (uchar)((double)std::rand() / (double)RAND_MAX * 255.0),
                           });
    }
}
