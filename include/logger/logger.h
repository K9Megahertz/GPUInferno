#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>

namespace CoreLogger {

    class StreamLogger;

    class Logger {
    public:
        enum class LogLevel {
            LOGLEVEL_DEBUG,
            LOGLEVEL_INFO,
            LOGLEVEL_WARN,
            LOGLEVEL_ERROR
        };

        Logger();
        ~Logger();

        StreamLogger Append(LogLevel level);

        std::string LogLevelAsString(LogLevel ll) const;

        void SetLevel(LogLevel level);
        void Write(const std::string& message);

        bool Start(const std::string& filename);
        void Stop();

        bool IsEnabled() const;

    private:
        LogLevel m_log_level;
        std::ofstream m_file;
        bool m_enabled;

        friend class StreamLogger;
    };

    class StreamLogger {
    public:
        StreamLogger(Logger* logger, Logger::LogLevel level, bool enabled)
            : m_logger(logger),
            m_level(level),
            m_enabled(enabled)
        {
        }

        StreamLogger(const StreamLogger&) = delete;
        StreamLogger& operator=(const StreamLogger&) = delete;

        StreamLogger(StreamLogger&&) = default;
        StreamLogger& operator=(StreamLogger&&) = default;

        ~StreamLogger();

        template <typename T>
        StreamLogger& operator<<(const T& value) {
            if (m_enabled) {
                m_stream << value;
            }
            return *this;
        }

        StreamLogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
            if (m_enabled) {
                manip(m_stream);
            }
            return *this;
        }

    private:
        Logger* m_logger;
        std::ostringstream m_stream;
        Logger::LogLevel m_level;
        bool m_enabled;
    };

}