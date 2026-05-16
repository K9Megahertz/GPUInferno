#pragma once

#include <inferno/util/logging.h>

namespace Inferno {

	CoreLogger::StreamLogger Append(Logger::LogLevel level);

}


#define INFERNO_LOG_DEBUG() \
    Inferno::Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG)

#define INFERNO_LOG_INFO() \
    Inferno::Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO)

#define INFERNO_LOG_WARN() \
    Inferno::Append(Inferno::Logger::LogLevel::LOGLEVEL_WARN)

#define INFERNO_LOG_ERROR() \
    Inferno::Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)