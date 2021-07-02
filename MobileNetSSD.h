#ifndef MOBILENETSSD_H
#define MOBILENETSSD_H

#include "MobileNetSSDGlobal.h"
#include "Process/OpenCV/dnn/COcvDnnProcess.h"
#include "Widget/OpenCV/dnn/COcvWidgetDnnCore.h"
#include "CPluginProcessInterface.hpp"

//------------------------------//
//----- CMobileNetSSDParam -----//
//------------------------------//
class MOBILENETSSDSHARED_EXPORT CMobileNetSSDParam: public COcvDnnProcessParam
{
    public:

        CMobileNetSSDParam() : COcvDnnProcessParam()
        {
            m_framework = Framework::CAFFE;
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            COcvDnnProcessParam::setParamMap(paramMap);
            m_confidence = std::stod(paramMap.at("confidence"));
            m_nmsThreshold = std::stod(paramMap.at("nmsThreshold"));
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = COcvDnnProcessParam::getParamMap();
            paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
            paramMap.insert(std::make_pair("nmsThreshold", std::to_string(m_nmsThreshold)));
            return paramMap;
        }

    public:

        double m_confidence = 0.5;
        double m_nmsThreshold = 0.4;
};

//-------------------------//
//----- CMobileNetSSD -----//
//-------------------------//
class MOBILENETSSDSHARED_EXPORT CMobileNetSSD: public COcvDnnProcess
{
    public:

        CMobileNetSSD();
        CMobileNetSSD(const std::string& name, const std::shared_ptr<CMobileNetSSDParam>& pParam);

        size_t      getProgressSteps() override;
        int         getNetworkInputSize() const override;
        double      getNetworkInputScaleFactor() const override;
        cv::Scalar  getNetworkInputMean() const override;

        void        run() override;

    private:

        void        manageOutput(cv::Mat &dnnOutput);
};

//--------------------------------//
//----- CMobileNetSSDFactory -----//
//--------------------------------//
class MOBILENETSSDSHARED_EXPORT CMobileNetSSDFactory : public CTaskFactory
{
    public:

        CMobileNetSSDFactory()
        {
            m_info.m_name = QObject::tr("MobileNet SSD").toStdString();
            m_info.m_shortDescription = QObject::tr("Single Shot Detector (SSD) for mobile and embedded vision applications.").toStdString();
            m_info.m_description = QObject::tr("We present a class of efficient models called MobileNets for mobile and embedded vision applications. "
                                               "MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. "
                                               "We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. "
                                               "These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. "
                                               "We present extensive experiments on resource and accuracy tradeoffs and "
                                               "show strong performance compared to other popular models on ImageNet classification. "
                                               "We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, "
                                               "finegrain classification, face attributes and large scale geo-localization.").toStdString();

            m_info.m_path = QObject::tr("Plugins/C++/Object/Detection").toStdString();
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_version = "1.0.0";
            m_info.m_authors = "Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam";
            m_info.m_article = "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications";
            m_info.m_year = 2017;
            m_info.m_license = "MIT License";
            m_info.m_repo = "https://github.com/chuanqi305/MobileNet-SSD";
            m_info.m_keywords = "deep,learning,detection,caffe,embedded";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CMobileNetSSDParam>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CMobileNetSSD>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CMobileNetSSDParam>();
            assert(paramPtr != nullptr);
            return std::make_shared<CMobileNetSSD>(m_info.m_name, paramPtr);
        }
};

//-------------------------------//
//----- CMobileNetSSDWidget -----//
//-------------------------------//
class MOBILENETSSDSHARED_EXPORT CMobileNetSSDWidget: public COcvWidgetDnnCore
{
    public:

        CMobileNetSSDWidget(QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(parent)
        {
            init();
        }
        CMobileNetSSDWidget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(parent)
        {
            m_pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(pParam);
            init();
        }

    private:

        void init() override
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CMobileNetSSDParam>();

            auto pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(m_pParam);
            assert(pParam);

            auto pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
            auto pSpinNmsThreshold = addDoubleSpin(tr("NMS threshold"), pParam->m_nmsThreshold, 0.0, 1.0, 0.1, 2);
            
            //Connections
            connect(pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(m_pParam);
                assert(pParam);
                pParam->m_confidence = val;
            });
            connect(pSpinNmsThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CMobileNetSSDParam>(m_pParam);
                assert(pParam);
                pParam->m_nmsThreshold = val;
            });
            connect(m_pApplyBtn, &QPushButton::clicked, [&]
            {
                emit doApplyProcess(m_pParam);
            });
        }
};

//--------------------------------------//
//----- CMobileNetSSDWidgetFactory -----//
//--------------------------------------//
class MOBILENETSSDSHARED_EXPORT CMobileNetSSDWidgetFactory : public CWidgetFactory
{
    public:

        CMobileNetSSDWidgetFactory()
        {
            m_name = QObject::tr("MobileNet SSD").toStdString();
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CMobileNetSSDWidget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class MOBILENETSSDSHARED_EXPORT CMobileNetSSDInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CMobileNetSSDFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CMobileNetSSDWidgetFactory>();
        }
};

#endif // MOBILENETSSD_H
