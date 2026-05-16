#pragma once

#include <Logger/Logger.h>

namespace Inferno {

    class Logger {
    public:
        using LogLevel = CoreLogger::Logger::LogLevel;

        static void SetLogger(CoreLogger::Logger* logger);

        static void EnableLogging();
        static void DisableLogging();

        //static void SetLevel(LogLevel level);

        

    private:
        static bool s_enabled;
        static LogLevel s_level;
        static CoreLogger::Logger* s_logger;

        friend CoreLogger::StreamLogger Append(LogLevel level);
    };

}