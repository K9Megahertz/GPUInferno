#include <Logger/Logger.h>
#include <filesystem>
#include <iostream>

namespace CoreLogger {

    Logger::Logger()
        : m_log_level(LogLevel::LOGLEVEL_INFO),
        m_enabled(false)
    {
    }

    Logger::~Logger() {
        Stop();
    }

    StreamLogger::~StreamLogger() {
        if (!m_enabled || m_logger == nullptr) {
            return;
        }

        std::time_t now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now()
        );

        std::tm local{};

#ifdef _WIN32
        localtime_s(&local, &now);
#else
        local = *std::localtime(&now);
#endif

        std::ostringstream final;

        final << "["
            << std::put_time(&local, "%F %T : ")
            << m_logger->LogLevelAsString(m_level)
            << "] "
            << m_stream.str();

        std::cout << final.str();
        std::cout.flush();

        m_logger->Write(final.str());
    }

    StreamLogger Logger::Append(Logger::LogLevel level) {
        bool should_log = m_enabled && level >= m_log_level;
        return StreamLogger(this, level, should_log);
    }

    std::string Logger::LogLevelAsString(Logger::LogLevel ll) const {
        switch (ll) {
        case LogLevel::LOGLEVEL_INFO:  return "INFO";
        case LogLevel::LOGLEVEL_DEBUG: return "DEBUG";
        case LogLevel::LOGLEVEL_ERROR: return "ERROR";
        case LogLevel::LOGLEVEL_WARN:  return "WARNING";
        default:                      return "UNKNOWN";
        }
    }

    void Logger::SetLevel(Logger::LogLevel level) {
        m_log_level = level;
    }

    void Logger::Write(const std::string& message) {
        if (m_file.is_open()) {
            m_file << message;
            m_file.flush();
        }
    }

    bool Logger::Start(const std::string& filename) {
        namespace fs = std::filesystem;

        fs::path path(filename);

        if (path.has_parent_path()) {
            fs::create_directories(path.parent_path());
        }

        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

        std::tm local{};

#ifdef _WIN32
        localtime_s(&local, &time);
#else
        local = *std::localtime(&time);
#endif

        std::ostringstream ss;
        ss << std::put_time(&local, "%Y-%m-%d.%H%M%S");

        fs::path finalpath =
            path.parent_path() /
            (path.stem().string() + "-" + ss.str() + path.extension().string());

        if (!finalpath.has_extension()) {
            finalpath += ".txt";
        }

        if (m_file.is_open()) {
            m_file.close();
        }

        m_file.clear();
        m_file.open(finalpath, std::ios::out | std::ios::app);

        if (!m_file.is_open()) {
            std::cerr << "Failed to open log file: " << finalpath << std::endl;
            m_enabled = false;
            return false;
        }

        m_enabled = true;

        Append(LogLevel::LOGLEVEL_INFO) << "*********************************" << std::endl;
        Append(LogLevel::LOGLEVEL_INFO) << "*        Logging Started        *" << std::endl;
        Append(LogLevel::LOGLEVEL_INFO) << "*********************************" << std::endl;
        Append(LogLevel::LOGLEVEL_INFO) << std::endl;
        Append(LogLevel::LOGLEVEL_INFO) << std::endl;

        return true;
    }

    void Logger::Stop() {
        m_enabled = false;

        if (m_file.is_open()) {
            m_file.close();
        }
    }

    bool Logger::IsEnabled() const {
        return m_enabled;
    }

}